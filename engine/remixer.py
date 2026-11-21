"""engine/remixer.py

Ghép vocal của Target Song lên beat của Reference Remix:
  1. Khớp tempo (time-stretch) và tông (pitch-shift) theo phân tích của Reference.
  2. Đặt đoạn vocal cao trào đúng vào điểm Drop của Reference.
  3. Reverb + Delay đồng bộ nhịp, sidechain theo từng cú kick thật của Reference.
  4. Cân bằng vocal/beat theo tỉ lệ của Reference, rồi Matchering để sao chép
     chất âm và độ lớn (LUFS) của bài mẫu.

Quy ước mảng audio trong module này: float32, shape (n_samples, 2), 44.1 kHz.
"""
from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal as sps

from .analyzer import (
    SR_PROC,
    STAGE_MASTER,
    STAGE_MATCH,
    ProgressFn,
    TrackAnalysis,
    TrackBundle,
    default_sidechain_curve,
    noop_progress,
    read_stereo,
)


# --------------------------------------------------------------------------- #
# Tham số & kế hoạch ghép
# --------------------------------------------------------------------------- #
@dataclass
class RemixParams:
    vocal_volume: float = 1.0       # 0..2, nhân thêm lên mức vocal tự cân bằng
    beat_volume: float = 1.0        # 0..2
    sidechain_depth: float = 0.8    # 0..1, độ sâu nén theo kick (chủ yếu trên reverb/delay)
    reverb_mix: float = 0.25        # 0..1, lượng reverb + delay
    anchor_time: Optional[float] = None    # giây trong Target; None = tự tìm đoạn cao trào
    semitone_shift: Optional[int] = None   # None = tự tính theo Key
    master: bool = True             # False = bỏ Matchering (nghe nhanh khi chỉnh mix)

    def clamp(self) -> "RemixParams":
        return RemixParams(
            vocal_volume=float(np.clip(self.vocal_volume, 0.0, 2.0)),
            beat_volume=float(np.clip(self.beat_volume, 0.0, 2.0)),
            sidechain_depth=float(np.clip(self.sidechain_depth, 0.0, 1.0)),
            reverb_mix=float(np.clip(self.reverb_mix, 0.0, 1.0)),
            anchor_time=None if self.anchor_time is None else max(0.0, float(self.anchor_time)),
            semitone_shift=None if self.semitone_shift is None else int(self.semitone_shift),
            master=bool(self.master),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "RemixParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names and v is not None}).clamp()


@dataclass
class MatchPlan:
    tempo_multiplier: float   # 0.5 / 1 / 2: chọn nhịp "hiệu dụng" của Reference gần Target nhất
    target_bpm: float
    ref_bpm: float
    stretch_factor: float     # độ dài sau / độ dài trước
    semitones: int
    anchor_src: float         # giây trong Target: đầu đoạn cao trào (đã căn vào lưới ô nhịp)
    drop_time: float          # giây trong Reference
    src_start: float          # đoạn Target cần xử lý [src_start, src_end]
    src_end: float
    place_time: float         # giây trong Reference nơi đặt src_start sau khi co giãn


def choose_tempo_multiplier(target_bpm: float, ref_bpm: float) -> float:
    return min((0.5, 1.0, 2.0), key=lambda m: abs(math.log(ref_bpm * m / target_bpm)))


def _relative_major(tonic: int, mode: str) -> int:
    return tonic if mode == "major" else (tonic + 3) % 12


def compute_semitone_shift(src_tonic: int, src_mode: str, dst_tonic: int, dst_mode: str) -> int:
    """Số bán cung nhỏ nhất đưa thang âm Target về thang âm Reference (so qua giọng trưởng tương đối)."""
    diff = (_relative_major(dst_tonic, dst_mode) - _relative_major(src_tonic, src_mode)) % 12
    return diff - 12 if diff > 6 else diff


def build_plan(ref: TrackAnalysis, tgt: TrackAnalysis, params: RemixParams) -> MatchPlan:
    if not tgt.has_vocals:
        raise ValueError("Target Song không có vocal đủ rõ để ghép vào bản remix.")

    mult = choose_tempo_multiplier(tgt.bpm, ref.bpm)
    stretch = tgt.bpm / (ref.bpm * mult)
    semitones = params.semitone_shift
    if semitones is None:
        semitones = compute_semitone_shift(tgt.key_tonic, tgt.key_mode, ref.key_tonic, ref.key_mode)

    bar_src = 4.0 * tgt.beat_period
    anchor = tgt.hook_start if params.anchor_time is None else params.anchor_time
    k = max(0, round((anchor - tgt.bar_offset) / bar_src))
    anchor = tgt.bar_offset + k * bar_src

    drop = ref.drop_time
    if drop >= ref.duration - 2.0:
        raise ValueError("Điểm Drop của Reference quá sát cuối bài để đặt vocal.")

    place0 = drop - anchor * stretch
    if place0 >= 0.0:
        src_start, place = 0.0, place0
    else:
        bars_before = int(math.floor(drop / (bar_src * stretch) + 1e-9))
        src_start = anchor - bars_before * bar_src
        place = drop - bars_before * bar_src * stretch
    src_end = min(tgt.duration, anchor + (ref.duration - drop) / stretch + 0.5)

    return MatchPlan(
        tempo_multiplier=float(mult),
        target_bpm=float(tgt.bpm),
        ref_bpm=float(ref.bpm),
        stretch_factor=float(stretch),
        semitones=int(semitones),
        anchor_src=float(anchor),
        drop_time=float(drop),
        src_start=float(src_start),
        src_end=float(src_end),
        place_time=float(place),
    )


# --------------------------------------------------------------------------- #
# Time-stretch + Pitch-shift (pyrubberband, fallback librosa)
# --------------------------------------------------------------------------- #
def rubberband_available() -> bool:
    return shutil.which("rubberband") is not None


def stretch_and_shift(y: np.ndarray, sr: int, rate: float, n_steps: int) -> tuple[np.ndarray, str]:
    """rate > 1 = nhanh hơn (ngắn hơn). Trả về (audio, tên engine đã dùng)."""
    no_stretch = abs(rate - 1.0) < 1e-6
    if no_stretch and n_steps == 0:
        return y, "none"

    if rubberband_available():
        try:
            import pyrubberband as pyrb

            rb_args: dict = {}
            if n_steps != 0:
                # pyrubberband chỉ nhận cặp (cờ, giá trị). Giá trị "-q" (quiet) vô hại,
                # dùng để bật cờ --formant giúp giữ nguyên chất giọng khi đổi tông.
                rb_args["--formant"] = "-q"
            if no_stretch:
                out = pyrb.pitch_shift(y, sr, n_steps, rbargs=rb_args)
            else:
                if n_steps != 0:
                    rb_args["--pitch"] = n_steps
                out = pyrb.time_stretch(y, sr, rate, rbargs=rb_args)
            return np.ascontiguousarray(np.asarray(out, dtype=np.float32)), "rubberband"
        except Exception:
            pass  # rơi xuống librosa

    import librosa

    channels = []
    for c in range(y.shape[1]):
        x = y[:, c]
        if not no_stretch:
            x = librosa.effects.time_stretch(x, rate=rate)
        if n_steps != 0:
            x = librosa.effects.pitch_shift(x, sr=sr, n_steps=n_steps)
        channels.append(x)
    n = min(len(c) for c in channels)
    out = np.stack([c[:n] for c in channels], axis=1).astype(np.float32)
    return out, "librosa"


def prepare_vocal(target: TrackBundle, plan: MatchPlan, ref_hash: str) -> tuple[np.ndarray, str]:
    """Vocal đã co giãn + đổi tông, cache theo (Reference, tempo, tông, đoạn cắt)."""
    key = (f"{ref_hash}|{plan.stretch_factor:.6f}|{plan.semitones}|"
           f"{plan.src_start:.3f}|{plan.src_end:.3f}")
    cache_path = target.root / f"vocal_{hashlib.sha1(key.encode()).hexdigest()[:16]}.wav"
    if cache_path.exists():
        return read_stereo(cache_path), "cache"

    vocals = read_stereo(target.stem_path("vocals"))
    seg = vocals[int(plan.src_start * SR_PROC):int(plan.src_end * SR_PROC)]
    out, engine = stretch_and_shift(seg, SR_PROC, 1.0 / plan.stretch_factor, plan.semitones)

    tmp = cache_path.with_name(cache_path.name + ".part")
    sf.write(str(tmp), out, SR_PROC, subtype="FLOAT", format="WAV")
    tmp.replace(cache_path)
    return out, engine


# --------------------------------------------------------------------------- #
# DSP: EQ, compressor, reverb, delay, sidechain
# --------------------------------------------------------------------------- #
def _sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sps.sosfilt(sos, x, axis=0).astype(np.float32)


def _peaking_sos(f0: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b = [1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin]
    a = [1.0 + alpha / a_lin, -2.0 * cw, 1.0 - alpha / a_lin]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def _one_pole(x: np.ndarray, time_s: float, sr: int) -> np.ndarray:
    a = math.exp(-1.0 / (sr * time_s))
    return sps.lfilter([1.0 - a], [1.0, -a], x)


def _compress(x: np.ndarray, sr: int, threshold_db: float = -22.0, ratio: float = 3.0,
              attack_s: float = 0.006, release_s: float = 0.12) -> np.ndarray:
    """Compressor feed-forward nhẹ, attack nhanh / release chậm."""
    detector = _one_pole(np.max(np.abs(x), axis=1).astype(np.float64), 0.010, sr)
    level_db = 20.0 * np.log10(np.maximum(detector, 1e-6))
    reduction = np.maximum(level_db - threshold_db, 0.0) * (1.0 - 1.0 / ratio)
    fast = _one_pole(reduction, attack_s, sr)
    slow = _one_pole(fast, release_s, sr)
    gain = 10.0 ** (-np.maximum(fast, slow) / 20.0)
    return (x * gain[:, None]).astype(np.float32)


def _make_reverb_ir(sr: int, rt60: float = 2.2, predelay: float = 0.025) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((n, 2)) * np.exp(-6.907755 * t / rt60)[:, None]
    ir = sps.sosfilt(sps.butter(2, 9000.0, "low", fs=sr, output="sos"), ir, axis=0)
    dark = sps.sosfilt(sps.butter(2, 3000.0, "low", fs=sr, output="sos"), ir, axis=0)
    blend = np.clip(t / rt60, 0.0, 1.0)[:, None]
    ir = ir * (1.0 - blend) + dark * blend
    ir = np.vstack([np.zeros((int(predelay * sr), 2)), ir])
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True)) + 1e-9
    return ir.astype(np.float32)


def _reverb(send: np.ndarray, sr: int) -> np.ndarray:
    ir = _make_reverb_ir(sr)
    n = len(send)
    return np.stack([sps.oaconvolve(send, ir[:, c])[:n] for c in range(2)], axis=1).astype(np.float32)


def _pingpong_delay(send: np.ndarray, sr: int, delay_s: float, feedback: float = 0.42,
                    repeats: int = 5) -> np.ndarray:
    n = len(send)
    out = np.zeros((n, 2), dtype=np.float32)
    lowpass = sps.butter(2, 4500.0, "low", fs=sr, output="sos")
    tap = send.astype(np.float32)
    step = int(round(delay_s * sr))
    for i in range(1, repeats + 1):
        offset = step * i
        if offset >= n:
            break
        tap = sps.sosfilt(lowpass, tap).astype(np.float32)
        out[offset:, (i - 1) % 2] += (0.7 * feedback ** (i - 1)) * tap[: n - offset]
    return out


def _duck_shape(curve, period: float, sr: int, attack_s: float = 0.006) -> np.ndarray:
    """Hình dạng nén 1 beat (0 = không nén, 1 = nén tối đa): hạ nhanh rồi hồi theo đường cong đo được."""
    c = np.asarray(curve, dtype=np.float64)
    c = c / max(c.max(), 1e-6)
    duck = (1.0 - c) / max(1.0 - c.min(), 1e-3)
    k = len(duck)
    peak = int(np.argmax(duck))
    if k - peak < 4:
        return _duck_shape(default_sidechain_curve(), period, sr, attack_s)
    recovery = np.minimum.accumulate(duck[peak:])
    recovery = (recovery - recovery[-1]) / max(recovery[0] - recovery[-1], 1e-6)

    length = max(int(period * sr), 1)
    tau = np.arange(length) / sr
    phase = peak / k + np.maximum(tau - attack_s, 0.0) / period
    shape = np.interp(phase, np.arange(peak, k) / k, recovery, right=0.0)
    shape = np.where(tau < attack_s, tau / attack_s, shape)
    return shape.astype(np.float32)


def build_duck_envelope(n: int, sr: int, ref: TrackAnalysis) -> np.ndarray:
    """Đường bao sidechain 0..1 kích hoạt bởi từng cú kick thật của Reference."""
    kicks = np.asarray(ref.kick_times, dtype=np.float64)
    if len(kicks) < 8:  # không tách được kick -> dùng lưới beat sau Drop
        count = int((n / sr - ref.drop_time) // ref.beat_period)
        kicks = ref.drop_time + np.arange(max(count, 0)) * ref.beat_period
    shape = _duck_shape(ref.sidechain_curve, ref.beat_period, sr)
    attack = int(0.006 * sr)
    duck = np.zeros(n, dtype=np.float32)
    for kick in np.sort(kicks):
        i0 = int(round(kick * sr))
        if i0 >= n:
            break
        i1 = min(n, i0 + len(shape))
        new = shape[: i1 - i0].copy()
        a = min(attack, len(new))
        new[:a] = np.maximum(new[:a], duck[i0:i0 + a])  # tránh nhảy bậc khi kick đến sớm
        duck[i0:i1] = new
    return duck


# --------------------------------------------------------------------------- #
# Mix
# --------------------------------------------------------------------------- #
def _load_beat(ref: TrackBundle) -> np.ndarray:
    acc: Optional[np.ndarray] = None
    for name in ("drums", "bass", "other"):
        stem = read_stereo(ref.stem_path(name))
        if acc is None:
            acc = stem
        else:
            n = min(len(acc), len(stem))
            acc = acc[:n] + stem[:n]
    assert acc is not None
    return acc


def _place_vocal(seg: np.ndarray, n: int, place_time: float, sr: int) -> np.ndarray:
    start = int(round(place_time * sr))
    if start >= n:
        raise ValueError("Vị trí đặt vocal nằm ngoài độ dài bản remix.")
    take = min(len(seg), n - start)
    chunk = seg[:take].copy()
    fade_in = min(int(0.015 * sr), take)
    if fade_in > 0:
        chunk[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    fade_out = min(int((0.8 if take < len(seg) else 0.03) * sr), take)
    if fade_out > 0:
        chunk[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    out = np.zeros((n, 2), dtype=np.float32)
    out[start:start + take] = chunk
    return out


def _active_levels(a: np.ndarray, b: np.ndarray) -> Optional[tuple[float, float]]:
    """Mức công suất (dB) của a và b, chỉ tính trên các frame mà a đang có tín hiệu."""
    frame = 4096
    n = min(len(a), len(b)) // frame * frame
    if n == 0:
        return None
    pa = np.mean(a[:n].astype(np.float64) ** 2, axis=1).reshape(-1, frame).mean(axis=1)
    pb = np.mean(b[:n].astype(np.float64) ** 2, axis=1).reshape(-1, frame).mean(axis=1)
    pa_db = 10.0 * np.log10(pa + 1e-12)
    mask = (pa_db > pa_db.max() - 30.0) & (pa_db > -55.0)
    if mask.sum() < 4:
        return None
    return (10.0 * np.log10(pa[mask].mean() + 1e-12), 10.0 * np.log10(pb[mask].mean() + 1e-12))


def _delay_seconds(period: float) -> float:
    delay = 0.75 * period   # nốt móc đơn chấm
    while delay < 0.2:
        delay *= 2.0
    while delay > 0.6:
        delay /= 2.0
    return delay


# --------------------------------------------------------------------------- #
# Mastering
# --------------------------------------------------------------------------- #
def _master_matchering(pre_path: Path, reference_wav: Path, out_path: Path,
                       progress: ProgressFn) -> None:
    import matchering as mg

    counter = {"n": 0}

    def on_log(message) -> None:
        counter["n"] += 1
        text = str(message).strip().replace("\n", " ")[:90]
        progress(STAGE_MASTER, min(0.95, 0.05 + 0.035 * counter["n"]), f"Matchering: {text}")

    mg.log(on_log)
    mg.process(target=str(pre_path), reference=str(reference_wav),
               results=[mg.pcm24(str(out_path))])


def _master_fallback(pre_path: Path, target_lufs: float, out_path: Path) -> None:
    data, sr = sf.read(str(pre_path), dtype="float64", always_2d=True)
    current = pyln.Meter(sr).integrated_loudness(data)
    if np.isfinite(current):
        data = data * 10.0 ** ((target_lufs - current) / 20.0)
    peak = float(np.max(np.abs(data)))
    if peak > 0.98:
        data *= 0.98 / peak
    sf.write(str(out_path), data, sr, subtype="PCM_24", format="WAV")


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render_remix(
    ref: TrackBundle,
    target: TrackBundle,
    params: RemixParams,
    out_path: str | Path,
    progress: ProgressFn = noop_progress,
) -> dict:
    """Render bản remix cuối ra out_path (WAV 24-bit). Không chạy lại Demucs/phân tích."""
    params = params.clamp()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ra, ta = ref.analysis, target.analysis
    sr = SR_PROC

    progress(STAGE_MATCH, 0.02, "Đang tính kế hoạch khớp nhịp, tông và vị trí Drop...")
    plan = build_plan(ra, ta, params)

    progress(STAGE_MATCH, 0.08,
             f"Đang khớp Vocal: {ta.bpm:.1f} → {ra.bpm * plan.tempo_multiplier:.1f} BPM, "
             f"đổi tông {plan.semitones:+d} bán cung...")
    vocal_seg, vocal_engine = prepare_vocal(target, plan, ref.file_hash)

    progress(STAGE_MATCH, 0.40, "Đang dựng beat từ stem của Reference...")
    beat = _load_beat(ref)
    n = len(beat)
    vocal = _place_vocal(vocal_seg, n, plan.place_time, sr)
    del vocal_seg

    progress(STAGE_MATCH, 0.50, "Đang xử lý vocal (EQ, compressor)...")
    v = _sos(vocal, sps.butter(2, 85.0, "high", fs=sr, output="sos"))
    v = _sos(v, _peaking_sos(300.0, -2.0, 1.0, sr))
    v = _sos(v, _peaking_sos(3200.0, 2.5, 0.9, sr))
    v = _compress(v, sr)
    del vocal

    # Cân bằng vocal/beat theo tỉ lệ đo được ở Reference (mặc định -3 dB)
    target_ratio = ra.vocal_to_beat_db if ra.vocal_to_beat_db is not None else -3.0
    target_ratio = float(np.clip(target_ratio, -10.0, 1.0))
    levels = _active_levels(v, beat)
    gain_db = 0.0 if levels is None else float(np.clip(target_ratio - (levels[0] - levels[1]), -18.0, 18.0))
    vocal_gain = 10.0 ** (gain_db / 20.0) * params.vocal_volume

    progress(STAGE_MATCH, 0.60, "Đang tạo Reverb / Delay đồng bộ nhịp...")
    mono = v.mean(axis=1)
    send = sps.sosfilt(sps.butter(2, [220.0, 9000.0], "bandpass", fs=sr, output="sos"), mono)
    wet = _reverb(send, sr) + 0.6 * _pingpong_delay(send, sr, _delay_seconds(ra.beat_period))
    del mono, send

    progress(STAGE_MATCH, 0.78, "Đang áp Sidechain theo nhịp Kick của Reference...")
    duck = build_duck_envelope(n, sr, ra)
    depth = params.sidechain_depth
    dry_gain = (1.0 - 0.3 * depth * duck)[:, None]
    wet_gain = (1.0 - depth * duck)[:, None]

    progress(STAGE_MATCH, 0.90, "Đang trộn Vocal + Beat...")
    mix = beat * params.beat_volume
    del beat
    mix += v * dry_gain * vocal_gain
    mix += wet * wet_gain * (vocal_gain * params.reverb_mix)
    del v, wet, duck

    peak = float(np.max(np.abs(mix)))
    if peak < 1e-6:
        raise ValueError("Bản mix rỗng: kiểm tra lại vị trí đặt vocal và mức âm lượng.")
    mix *= 10.0 ** (-3.0 / 20.0) / peak  # chừa headroom -3 dBFS trước khi master

    pre_path = out_path.with_name(out_path.stem + ".premaster.wav")
    sf.write(str(pre_path), mix, sr, subtype="PCM_24", format="WAV")
    del mix
    progress(STAGE_MATCH, 1.0, "Đã khớp xong Vocal và Beat")

    mastering_engine = "none"
    try:
        if params.master:
            progress(STAGE_MASTER, 0.02, "Đang Mastering theo chất âm của Reference...")
            try:
                _master_matchering(pre_path, ref.mix_path, out_path, progress)
                mastering_engine = "matchering"
            except Exception as exc:  # Matchering từ chối (file quá ngắn, thiếu thư viện...)
                _master_fallback(pre_path, ra.lufs, out_path)
                mastering_engine = f"loudness-fallback ({type(exc).__name__})"
            progress(STAGE_MASTER, 1.0, "Mastering hoàn tất")
        else:
            pre_path.replace(out_path)
    finally:
        pre_path.unlink(missing_ok=True)

    return {
        "output": str(out_path),
        "duration": n / sr,
        "plan": asdict(plan),
        "vocal_engine": vocal_engine,
        "vocal_gain_db": round(gain_db, 2),
        "mastering_engine": mastering_engine,
        "params": asdict(params),
    }
