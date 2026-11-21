"""engine/analyzer.py

Giải mã audio, tách stem bằng Demucs (htdemucs) và phân tích:
BPM + lưới nhịp, Key, điểm Drop, cấu trúc đoạn, đường cong Sidechain,
độ lớn (LUFS) và phổ tần số. Mọi kết quả được cache theo SHA-1 của file upload.

Cấu trúc cache:
    cache/<sha1>/mix.wav               44.1 kHz stereo 24-bit
    cache/<sha1>/stems/<stem>.flac     vocals / drums / bass / other (24-bit)
    cache/<sha1>/analysis.json         kết quả phân tích
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Optional

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal as sps
from scipy.ndimage import median_filter

# --------------------------------------------------------------------------- #
# Hằng số
# --------------------------------------------------------------------------- #
SR_PROC = 44100          # sample rate xử lý/ghi file (bằng sample rate của htdemucs)
SR_ANALYSIS = 22050      # sample rate dùng cho phân tích (nhanh hơn, đủ cho nhịp/key)
HOP = 256                # hop length cho các đường bao onset (~11.6 ms)
STEM_NAMES = ("vocals", "drums", "bass", "other")
ANALYSIS_VERSION = 1
SIDECHAIN_POINTS = 64

STAGE_DECODE = "decode"
STAGE_DEMUCS = "demucs"
STAGE_ANALYSIS = "analysis"
STAGE_MATCH = "matching"
STAGE_MASTER = "mastering"

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# progress(stage, fraction 0..1 trong stage, message)
ProgressFn = Callable[[str, float, str], None]


def noop_progress(stage: str, fraction: float, message: str) -> None:
    return None


# --------------------------------------------------------------------------- #
# Tiện ích nhỏ
# --------------------------------------------------------------------------- #
def _db(x, floor: float = 1e-12):
    return 10.0 * np.log10(np.maximum(x, floor))


def fmt_time(sec: float) -> str:
    minutes, seconds = divmod(max(float(sec), 0.0), 60.0)
    return f"{int(minutes)}:{seconds:06.3f}"


def _window_power(cs: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Công suất trung bình trong các cửa sổ [starts, ends) từ mảng cộng dồn cs."""
    starts = np.clip(starts, 0, len(cs) - 1)
    ends = np.clip(ends, 0, len(cs) - 1)
    width = np.maximum(ends - starts, 1)
    return (cs[ends] - cs[starts]) / width


def file_sha1(path: str | Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(path: str | Path) -> np.ndarray:
    """Đọc bất kỳ file audio nào -> float32 (n, 2) ở 44.1 kHz."""
    y, _ = librosa.load(str(path), sr=SR_PROC, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    else:
        y = y[:2]
    y = np.clip(y, -1.0, 1.0)
    return np.ascontiguousarray(y.T.astype(np.float32))


def read_stereo(path: str | Path) -> np.ndarray:
    """Đọc file đã cache (44.1 kHz) -> float32 (n, 2)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR_PROC:
        raise RuntimeError(f"File cache {path} có sample rate {sr}, kỳ vọng {SR_PROC}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return np.ascontiguousarray(data[:, :2])


def _load_mono(path: str | Path, sr: int = SR_ANALYSIS) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _write_atomic(path: Path, data: np.ndarray, sr: int, subtype: str, fmt: str) -> None:
    tmp = path.with_name(path.name + ".part")
    sf.write(str(tmp), data, sr, subtype=subtype, format=fmt)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Demucs
# --------------------------------------------------------------------------- #
_DEMUCS: Optional[tuple] = None
_DEMUCS_LOCK = threading.Lock()


def _get_demucs():
    global _DEMUCS
    with _DEMUCS_LOCK:
        if _DEMUCS is None:
            import torch
            from demucs.pretrained import get_model

            model = get_model("htdemucs")
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            _DEMUCS = (model, device)
        return _DEMUCS


def separate_stems(
    audio: np.ndarray,
    stems_dir: Path,
    progress: ProgressFn = noop_progress,
    prefix: str = "",
) -> None:
    """Tách 4 stem bằng htdemucs. audio: float32 (n, 2) @44.1 kHz.

    Xử lý theo từng khúc 30 s (chồng 2 s, crossfade tuyến tính) để có tiến độ
    thực tế và giữ RAM ổn định với bài dài.
    """
    import torch
    from demucs.apply import apply_model

    model, device = _get_demucs()
    sources = list(model.sources)
    wav = np.ascontiguousarray(audio.T)  # (2, n)
    n = wav.shape[1]
    chunk, overlap = 30 * SR_PROC, 2 * SR_PROC
    step = chunk - overlap

    starts, s = [], 0
    while True:
        starts.append(s)
        if s + chunk >= n:
            break
        s += step

    out = np.zeros((len(sources), 2, n), dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    ramp_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    ramp_out = ramp_in[::-1]

    for i, start in enumerate(starts):
        end = min(start + chunk, n)
        seg = torch.from_numpy(np.ascontiguousarray(wav[:, start:end]))
        ref = seg.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            pred = apply_model(
                model, ((seg - mean) / std)[None],
                shifts=1, split=True, overlap=0.25, progress=False, device=device,
            )[0]
        pred = (pred * std + mean).cpu().numpy()

        w = np.ones(end - start, dtype=np.float32)
        if i > 0:
            w[:overlap] *= ramp_in
        if end < n:
            w[-overlap:] *= ramp_out
        out[:, :, start:end] += pred * w
        weight[start:end] += w
        progress(STAGE_DEMUCS, (i + 1) / len(starts),
                 f"{prefix}Đang tách stem bằng Demucs ({i + 1}/{len(starts)})")

    out /= np.maximum(weight, 1e-6)[None, None, :]
    stems_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(sources):
        data = np.clip(out[idx].T, -1.0, 1.0)
        _write_atomic(stems_dir / f"{name}.flac", data, SR_PROC, "PCM_24", "FLAC")

    del out, weight
    if device == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Kết quả phân tích
# --------------------------------------------------------------------------- #
@dataclass
class TrackAnalysis:
    version: int
    file_hash: str
    duration: float
    bpm: float
    beat_period: float          # giây / beat (BPM đã gấp về khoảng 80-170)
    beat_offset: float          # thời điểm beat lưới đầu tiên, nằm trong [0, beat_period)
    bar_offset: float           # thời điểm vạch nhịp (beat 1) đầu tiên, nằm trong [0, 4*beat_period)
    key_tonic: int
    key_mode: str               # "major" | "minor"
    key_name: str
    key_confidence: float
    drop_time: float
    drop_score_db: float
    drops: list
    sections: list
    hook_start: float           # đầu đoạn vocal "cao trào" (điểm neo để đặt vào Drop)
    has_vocals: bool
    vocal_to_beat_db: Optional[float]
    kick_times: list
    sidechain_curve: list       # độ lợi 0..1 trong 1 beat kể từ lúc kick (1 = không nén)
    sidechain_depth_db: float
    sidechain_detected: bool
    lufs: float
    peak_db: float
    spectral_centroid: float
    spectrum_freqs: list
    spectrum_db: list

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrackAnalysis":
        names = [f.name for f in fields(cls)]
        return cls(**{name: data[name] for name in names})

    def summary(self) -> dict:
        """Bản rút gọn gửi cho giao diện web."""
        return {
            "duration": round(self.duration, 2),
            "bpm": round(self.bpm, 2),
            "key": self.key_name,
            "key_confidence": round(self.key_confidence, 3),
            "drop_time": round(self.drop_time, 3),
            "drop_label": fmt_time(self.drop_time),
            "drop_score_db": round(self.drop_score_db, 2),
            "drops": [
                {"time": round(d["time"], 3), "label": fmt_time(d["time"]),
                 "score_db": round(d["score_db"], 2)}
                for d in self.drops
            ],
            "sections": self.sections,
            "hook_time": round(self.hook_start, 3),
            "hook_label": fmt_time(self.hook_start),
            "has_vocals": self.has_vocals,
            "lufs": round(self.lufs, 1),
            "peak_db": round(self.peak_db, 1),
            "sidechain_depth_db": round(self.sidechain_depth_db, 1),
            "sidechain_detected": self.sidechain_detected,
            "spectral_centroid": round(self.spectral_centroid, 1),
            "spectrum": {"freqs": self.spectrum_freqs, "db": self.spectrum_db},
        }


@dataclass
class TrackBundle:
    """Trọn bộ tài nguyên đã cache của 1 bài nhạc."""
    file_hash: str
    root: Path
    analysis: TrackAnalysis

    @property
    def mix_path(self) -> Path:
        return self.root / "mix.wav"

    @property
    def stems_dir(self) -> Path:
        return self.root / "stems"

    def stem_path(self, name: str) -> Path:
        return self.stems_dir / f"{name}.flac"


# --------------------------------------------------------------------------- #
# Nhịp: BPM, lưới beat, Drop
# --------------------------------------------------------------------------- #
def _low_onset_env(y: np.ndarray, fmax: float = 150.0) -> np.ndarray:
    """Đường bao onset của dải trầm 30-150 Hz (kick / bass), chuẩn hóa về max = 1."""
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR_ANALYSIS, n_fft=2048)
    band = spec[(freqs >= 30.0) & (freqs <= fmax)]
    logmag = np.log1p(30.0 * band)
    flux = np.maximum(0.0, np.diff(logmag, axis=1, prepend=logmag[:, :1])).sum(axis=0)
    return flux / (flux.max() + 1e-9)


def _fit_beat_grid(beats: np.ndarray) -> tuple[float, float]:
    """Hồi quy tuyến tính beat_time = anchor + k * period, có loại điểm ngoại lai."""
    period0 = float(np.median(np.diff(beats)))
    idx = np.round((beats - beats[0]) / period0)
    keep = np.ones(len(beats), dtype=bool)
    slope, intercept = period0, float(beats[0])
    for _ in range(3):
        slope, intercept = np.polyfit(idx[keep], beats[keep], 1)
        resid = np.abs(beats - (slope * idx + intercept))
        new_keep = resid < 0.12 * slope
        if new_keep.sum() < 8 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return float(slope), float(intercept)


def _beat_grid(y_perc: np.ndarray, low_env: np.ndarray) -> tuple[float, float]:
    """Trả về (period, offset) của lưới beat, căn theo kick."""
    env = librosa.onset.onset_strength(y=y_perc, sr=SR_ANALYSIS, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=env, sr=SR_ANALYSIS, hop_length=HOP, units="time", tightness=100
    )
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) >= 8:
        period, anchor = _fit_beat_grid(beats)
    else:
        period = 60.0 / max(tempo, 1.0)
        anchor = float(beats[0]) if len(beats) else 0.0

    bpm = 60.0 / period
    while bpm < 80.0:
        period /= 2.0
        bpm *= 2.0
    while bpm >= 170.0:
        period *= 2.0
        bpm /= 2.0
    offset = anchor % period

    # Tinh chỉnh offset theo vị trí kick thực tế (trung bình pha vòng tròn)
    frame_t = librosa.frames_to_time(np.arange(len(low_env)), sr=SR_ANALYSIS, hop_length=HOP)
    thr = np.percentile(low_env, 75)
    sel = low_env > thr
    if sel.sum() > 16 and thr > 0:
        weights = low_env[sel]
        angle = 2.0 * np.pi * (((frame_t[sel] - offset) / period) % 1.0)
        z = np.sum(weights * np.exp(1j * angle))
        if abs(z) / np.sum(weights) > 0.3:
            offset = (offset + np.angle(z) / (2.0 * np.pi) * period) % period
    return float(period), float(offset)


def _refine_kicks(low_sig: np.ndarray, kicks: np.ndarray) -> np.ndarray:
    """Căn từng kick về điểm đường bao biên độ dốc nhất (khử độ trễ của cửa sổ STFT, ~10 ms)."""
    if len(kicks) == 0:
        return kicks
    win = max(1, int(0.008 * SR_ANALYSIS))
    env = np.convolve(np.abs(low_sig), np.ones(win) / win, mode="same")
    rise = np.diff(env, prepend=env[0])
    refined = []
    for kick in kicks:
        i0 = max(0, int((kick - 0.04) * SR_ANALYSIS))
        i1 = min(len(rise), int((kick + 0.05) * SR_ANALYSIS))
        if i1 - i0 < 4:
            refined.append(kick)
        else:
            refined.append((i0 + int(np.argmax(rise[i0:i1]))) / SR_ANALYSIS)
    return np.asarray(refined, dtype=np.float64)


def _align_offset(kicks: np.ndarray, period: float, offset: float) -> float:
    """Dời lưới beat về đúng tâm các cú kick đã tinh chỉnh (trung vị độ lệch pha)."""
    if len(kicks) < 8:
        return offset
    phase = ((kicks - offset) / period + 0.5) % 1.0 - 0.5
    near = np.abs(phase) < 0.2
    if near.sum() < 8:
        return offset
    return float((offset + np.median(phase[near]) * period) % period)


def _beat_energy(low_sig: np.ndarray, period: float, offset: float) -> np.ndarray:
    """Năng lượng dải trầm (dB) của từng beat theo lưới."""
    n_beats = int((len(low_sig) / SR_ANALYSIS - offset) // period)
    k = np.arange(n_beats + 1)
    idx = np.round((offset + k * period) * SR_ANALYSIS).astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(low_sig.astype(np.float64) ** 2)])
    return np.maximum(_db(_window_power(cs, idx[:-1], idx[1:])), -80.0)


def _detect_drops(energy: np.ndarray, period: float, offset: float) -> tuple[list, int, float]:
    """Drop = beat mà năng lượng trầm của 8 beat sau vượt xa 8 beat trước nó.

    Score(k) = mean(E[k:k+8]) - mean(E[k-8:k]) đạt cực đại đúng tại beat đầu tiên
    của đoạn cao trào. Chọn Drop chính = ứng viên sớm nhất có score >= 70% score lớn nhất.
    """
    n, w = len(energy), 8
    if n < 2 * w + 1:
        return [], 0, 0.0
    cs = np.concatenate([[0.0], np.cumsum(energy)])
    k = np.arange(w, n - w + 1)
    score = np.zeros(n)
    score[k] = (cs[k + w] - cs[k]) / w - (cs[k] - cs[k - w]) / w

    peaks, _ = sps.find_peaks(score, height=max(3.0, 0.35 * score.max()), distance=16)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(score))])
    best = float(score[peaks].max())
    primary = int(next(p for p in peaks if score[p] >= 0.7 * best))

    top = sorted(peaks, key=lambda p: score[p], reverse=True)[:6]
    drops = [
        {"time": float(offset + p * period), "beat": int(p), "score_db": float(score[p])}
        for p in sorted(top)
    ]
    return drops, primary, float(score[primary])


def _sections(energy: np.ndarray, first_beat: int, period: float, offset: float,
              duration: float) -> list:
    """Phân đoạn thô: intro / breakdown / drop / outro dựa trên năng lượng trầm theo ô nhịp."""
    n_bars = (len(energy) - first_beat) // 4
    if n_bars < 4:
        return [{"label": "drop", "start": 0.0, "end": float(duration)}]
    bar_e = energy[first_beat:first_beat + 4 * n_bars].reshape(n_bars, 4).mean(axis=1)
    low, high = np.percentile(bar_e, [15, 85])
    if high - low < 3.0:
        return [{"label": "drop", "start": 0.0, "end": float(duration)}]
    state = (bar_e > high - 6.0).astype(int)
    state = median_filter(state, size=3, mode="nearest")

    sections, a = [], 0
    for b in range(1, n_bars + 1):
        if b == n_bars or state[b] != state[a]:
            if state[a] == 1:
                label = "drop"
            elif a == 0:
                label = "intro"
            elif b == n_bars:
                label = "outro"
            else:
                label = "breakdown"
            start = 0.0 if a == 0 else offset + (first_beat + 4 * a) * period
            end = duration if b == n_bars else offset + (first_beat + 4 * b) * period
            sections.append({"label": label, "start": round(float(start), 3),
                             "end": round(float(end), 3)})
            a = b
    return sections


# --------------------------------------------------------------------------- #
# Sidechain
# --------------------------------------------------------------------------- #
def default_sidechain_curve() -> np.ndarray:
    phase = np.arange(SIDECHAIN_POINTS) / SIDECHAIN_POINTS
    return 1.0 - 0.85 * np.exp(-phase / 0.18)


def _estimate_sidechain(src: np.ndarray, kicks: np.ndarray, period: float) -> Optional[np.ndarray]:
    """Gấp đường bao RMS của nhạc cụ nền (bị nén bởi sidechain) theo thời điểm kick."""
    frame_len = 512
    rms = librosa.feature.rms(y=src, frame_length=frame_len, hop_length=HOP)[0]
    # Gán nhãn mỗi frame theo thời điểm BẮT ĐẦU cửa sổ để frame đầu tiên sau kick không lẫn phần trước kick
    frame_t = (librosa.frames_to_time(np.arange(len(rms)), sr=SR_ANALYSIS, hop_length=HOP)
               - 0.5 * frame_len / SR_ANALYSIS)
    grid = np.arange(SIDECHAIN_POINTS) / SIDECHAIN_POINTS
    acc, count = np.zeros(SIDECHAIN_POINTS), 0
    for kick in kicks:
        i0 = int(np.searchsorted(frame_t, kick))
        i1 = int(np.searchsorted(frame_t, kick + period))
        seg = rms[i0:i1]
        if len(seg) < 8 or seg.max() < 1e-4:
            continue
        phase = (frame_t[i0:i1] - kick) / period
        acc += np.interp(grid, phase, seg / seg.max())
        count += 1
    if count < 8:
        return None
    curve = acc / count
    curve /= max(curve.max(), 1e-9)
    padded = np.pad(curve, (2, 2), mode="edge")
    return np.convolve(padded, np.ones(5) / 5.0, mode="valid")


# --------------------------------------------------------------------------- #
# Key, hook vocal, loudness / spectrum
# --------------------------------------------------------------------------- #
def _detect_key(y: np.ndarray) -> tuple[int, str, float]:
    y_low = librosa.resample(y, orig_sr=SR_ANALYSIS, target_sr=SR_ANALYSIS // 2)
    chroma = librosa.feature.chroma_cqt(
        y=y_low, sr=SR_ANALYSIS // 2, hop_length=2048, bins_per_octave=36, n_octaves=6
    )
    rms = librosa.feature.rms(y=y_low, hop_length=2048)[0]
    frames = min(chroma.shape[1], len(rms))
    weights = rms[:frames]
    profile = (chroma[:, :frames] * weights).sum(axis=1) / (weights.sum() + 1e-9)

    scores = []
    for tonic in range(12):
        for mode, template in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
            corr = np.corrcoef(profile, np.roll(template, tonic))[0, 1]
            scores.append((float(np.nan_to_num(corr)), tonic, mode))
    scores.sort(reverse=True)
    best_corr, tonic, mode = scores[0]
    return tonic, mode, max(best_corr, 0.0)


def _find_hook(vocals: np.ndarray, bar_offset: float, period: float) -> float:
    """Đầu đoạn 8 ô nhịp vocal to nhất (ưu tiên đoạn sớm nhất gần ngang mức đó)."""
    bar_len = 4.0 * period
    total = len(vocals) / SR_ANALYSIS
    n_bars = int((total - bar_offset) // bar_len)
    if n_bars < 2:
        return 0.0
    bars = np.arange(n_bars)
    starts = np.round((bar_offset + bars * bar_len) * SR_ANALYSIS).astype(np.int64)
    ends = np.minimum(np.round((bar_offset + (bars + 1) * bar_len) * SR_ANALYSIS).astype(np.int64),
                      len(vocals))
    cs = np.concatenate([[0.0], np.cumsum(vocals.astype(np.float64) ** 2)])
    vdb = np.maximum(_db(_window_power(cs, starts, ends)), -60.0)
    w = min(8, n_bars)
    window = np.convolve(vdb, np.ones(w) / w, mode="valid")
    best = int(np.flatnonzero(window >= window.max() - 2.0)[0])
    start = best
    while start > 0 and best - start < 4 and vdb[start - 1] >= window[best] - 3.0:
        start -= 1
    return float(bar_offset + start * bar_len)


def _vocal_to_beat_db(vocals: np.ndarray, beat: np.ndarray) -> tuple[bool, Optional[float]]:
    """(có vocal?, chênh lệch công suất vocal/beat tính trên các frame vocal đang hát)."""
    frame = 2048
    n = min(len(vocals), len(beat)) // frame * frame
    if n == 0:
        return False, None
    pv = (vocals[:n].astype(np.float64) ** 2).reshape(-1, frame).mean(axis=1)
    pb = (beat[:n].astype(np.float64) ** 2).reshape(-1, frame).mean(axis=1)
    pv_db = _db(pv)
    if pv_db.max() < -45.0:
        return False, None
    mask = pv_db > pv_db.max() - 25.0
    if mask.sum() < 20:
        return True, None
    return True, float(_db(pv[mask].mean()) - _db(pb[mask].mean()))


def _loudness_and_spectrum(mix_path: Path):
    data, sr = sf.read(str(mix_path), dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    try:
        lufs = float(pyln.Meter(sr).integrated_loudness(data.astype(np.float64)))
        if not np.isfinite(lufs):
            lufs = -70.0
    except Exception:
        lufs = -70.0

    mono = data.mean(axis=1)
    freqs, psd = sps.welch(mono, fs=sr, nperseg=8192, noverlap=4096)
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-20))
    df = float(freqs[1] - freqs[0])
    centers = 1000.0 * 2.0 ** (np.arange(-16, 14) / 3.0)   # 1/3 octave, ~25 Hz -> 20 kHz
    levels = []
    for fc in centers:
        lo, hi = fc * 2.0 ** (-1.0 / 6.0), fc * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= lo) & (freqs < hi)
        power = psd[mask].sum() * df if mask.any() else float(np.interp(fc, freqs, psd)) * (hi - lo)
        levels.append(10.0 * np.log10(power + 1e-20))
    levels = np.array(levels)
    levels -= levels.max()
    return (
        lufs,
        float(20.0 * np.log10(max(peak, 1e-9))),
        centroid,
        [round(float(c), 1) for c in centers],
        [round(float(v), 2) for v in levels],
    )


# --------------------------------------------------------------------------- #
# Phân tích tổng hợp
# --------------------------------------------------------------------------- #
def analyze_track(mix_path: Path, stems_dir: Path, file_hash: str) -> TrackAnalysis:
    stems = {name: _load_mono(stems_dir / f"{name}.flac") for name in STEM_NAMES}
    duration = len(stems["drums"]) / SR_ANALYSIS
    beat_mix = stems["drums"] + stems["bass"] + stems["other"]

    drums_rms = float(np.sqrt(np.mean(stems["drums"] ** 2)))
    perc = stems["drums"] if drums_rms > 1e-4 else beat_mix
    low_env = _low_onset_env(perc)
    period, offset = _beat_grid(perc, low_env)
    bpm = 60.0 / period

    # Kick: phát hiện thô -> tinh chỉnh theo miền thời gian -> căn lại lưới beat
    sos_low = sps.butter(4, 150.0, btype="low", fs=SR_ANALYSIS, output="sos")
    kicks = librosa.onset.onset_detect(
        onset_envelope=low_env, sr=SR_ANALYSIS, hop_length=HOP, units="time",
        delta=0.08, wait=max(1, int(0.4 * period * SR_ANALYSIS / HOP)),
    )
    kicks = np.asarray(kicks, dtype=np.float64)
    kicks = _refine_kicks(sps.sosfiltfilt(sos_low, perc), kicks)
    offset = _align_offset(kicks, period, offset)

    # Drop dựa trên năng lượng dải trầm của drums + bass
    low_sig = sps.sosfiltfilt(sos_low, stems["drums"] + stems["bass"])
    energy = _beat_energy(low_sig, period, offset)
    drops, drop_beat, drop_score = _detect_drops(energy, period, offset)
    drop_time = offset + drop_beat * period
    first_beat = drop_beat % 4                    # Drop luôn là beat 1 của một ô nhịp
    bar_offset = offset + first_beat * period
    sections = _sections(energy, first_beat, period, offset, duration)

    # Sidechain: đo đường bao nhạc nền theo từng cú kick đúng nhịp
    if len(kicks):
        phase_err = np.abs(((kicks - offset) / period + 0.5) % 1.0 - 0.5)
        on_beat = kicks[phase_err < 0.15]
    else:
        on_beat = kicks
    chosen = on_beat[on_beat >= drop_time]
    if len(chosen) < 16:
        chosen = on_beat
    pump_src = stems["other"] if float(np.sqrt(np.mean(stems["other"] ** 2))) > 1e-3 \
        else stems["other"] + stems["bass"]
    curve = _estimate_sidechain(pump_src, chosen, period) if len(chosen) else None
    if curve is not None and float(curve.min()) < 0.9:
        sidechain_detected = True
    else:
        curve, sidechain_detected = default_sidechain_curve(), False
    depth_db = float(20.0 * np.log10(max(float(curve.min()), 1e-3)))

    # Key trên phần hòa âm (vocal + bass + other), bỏ drums
    tonic, mode, key_conf = _detect_key(stems["vocals"] + stems["bass"] + stems["other"])

    has_vocals, v2b = _vocal_to_beat_db(stems["vocals"], beat_mix)
    hook = _find_hook(stems["vocals"], bar_offset, period) if has_vocals else 0.0

    lufs, peak_db, centroid, sp_freqs, sp_db = _loudness_and_spectrum(mix_path)

    return TrackAnalysis(
        version=ANALYSIS_VERSION,
        file_hash=file_hash,
        duration=float(duration),
        bpm=float(bpm),
        beat_period=float(period),
        beat_offset=float(offset),
        bar_offset=float(bar_offset),
        key_tonic=int(tonic),
        key_mode=mode,
        key_name=f"{NOTE_NAMES[tonic]} {mode}",
        key_confidence=float(key_conf),
        drop_time=float(drop_time),
        drop_score_db=float(drop_score),
        drops=drops,
        sections=sections,
        hook_start=float(hook),
        has_vocals=bool(has_vocals),
        vocal_to_beat_db=None if v2b is None else float(v2b),
        kick_times=[round(float(t), 4) for t in kicks],
        sidechain_curve=[round(float(c), 4) for c in curve],
        sidechain_depth_db=depth_db,
        sidechain_detected=bool(sidechain_detected),
        lufs=lufs,
        peak_db=peak_db,
        spectral_centroid=centroid,
        spectrum_freqs=sp_freqs,
        spectrum_db=sp_db,
    )


# --------------------------------------------------------------------------- #
# Chuẩn bị 1 bài: decode -> stems -> analysis (có cache)
# --------------------------------------------------------------------------- #
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def prepare_track(
    upload_path: str | Path,
    cache_root: str | Path,
    progress: ProgressFn = noop_progress,
    label: str = "",
) -> TrackBundle:
    """Trả về TrackBundle; bước nào đã có trong cache sẽ được bỏ qua.

    label: tiền tố hiển thị trong thông báo tiến độ, ví dụ "Reference".
    """
    prefix = f"[{label}] " if label else ""
    upload_path = Path(upload_path)
    progress(STAGE_DECODE, 0.0, f"{prefix}Đang đọc file âm thanh...")
    file_hash = file_sha1(upload_path)
    root = Path(cache_root) / file_hash
    stems_dir = root / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    with _lock_for(file_hash):
        mix_path = root / "mix.wav"
        audio: Optional[np.ndarray] = None
        if not mix_path.exists():
            audio = decode_audio(upload_path)
            _write_atomic(mix_path, audio, SR_PROC, "PCM_24", "WAV")
        progress(STAGE_DECODE, 1.0, f"{prefix}Đã đọc xong file âm thanh")

        if not all((stems_dir / f"{name}.flac").exists() for name in STEM_NAMES):
            if audio is None:
                audio = read_stereo(mix_path)
            separate_stems(audio, stems_dir, progress, prefix)
        else:
            progress(STAGE_DEMUCS, 1.0, f"{prefix}Đã có stem trong cache, bỏ qua Demucs")
        del audio

        analysis_path = root / "analysis.json"
        analysis: Optional[TrackAnalysis] = None
        if analysis_path.exists():
            try:
                loaded = TrackAnalysis.from_dict(json.loads(analysis_path.read_text(encoding="utf-8")))
                if loaded.version == ANALYSIS_VERSION:
                    analysis = loaded
            except (KeyError, TypeError, ValueError):
                analysis = None
        if analysis is None:
            progress(STAGE_ANALYSIS, 0.0, f"{prefix}Đang phân tích nhịp, tông, Drop, Sidechain...")
            analysis = analyze_track(mix_path, stems_dir, file_hash)
            tmp = analysis_path.with_name(analysis_path.name + ".part")
            tmp.write_text(json.dumps(analysis.to_dict(), ensure_ascii=False), encoding="utf-8")
            tmp.replace(analysis_path)
        progress(STAGE_ANALYSIS, 1.0, f"{prefix}Phân tích hoàn tất")

    return TrackBundle(file_hash=file_hash, root=root, analysis=analysis)
