"""main.py - AI Remix Style Transfer Studio (FastAPI + Uvicorn).

Chạy:  python main.py   ->  http://localhost:8000

API (mọi tác vụ nặng chạy nền bằng BackgroundTasks, giao diện hỏi tiến độ qua /api/jobs/<id>):

    GET  /api/system                 Thông tin môi trường, giới hạn, danh sách bước tiến trình.
    POST /api/analyze/{role}         Upload file (form field "file"); role = reference | target.
                                     -> {"job_id"}. Job chạy: giải mã -> Demucs -> phân tích.
    GET  /api/jobs/{job_id}          Trạng thái + tiến độ + kết quả/lỗi của job.
    GET  /api/tracks/{hash}          Kết quả phân tích của 1 bài đã cache (nạp lại giao diện).
    GET  /api/tracks/{hash}/audio    Nghe lại bản gốc đã giải mã (hỗ trợ Range).
    POST /api/render                 JSON {"reference", "target", <RemixParams>} -> {"job_id"}.
                                     Không tách Demucs lại: dùng lại stem + phân tích trong cache.
    GET  /api/renders/{id}           Thông tin 1 bản remix đã render.
    GET  /api/renders/{id}/audio     Phát bản remix (cho Wavesurfer, hỗ trợ Range).
    GET  /api/renders/{id}/download  Tải .wav (Content-Disposition: attachment).

Thiết kế:
  * Một khóa chung cho mọi job nặng: Demucs giữ model dùng chung và Matchering dùng logger toàn
    cục (mg.log), nên các job xếp hàng và chạy lần lượt (job đang chờ báo status = "queued").
  * Cache theo SHA-1 nằm hoàn toàn trong engine (cache/<sha1>/). Server không giữ audio trong RAM;
    mỗi lần render chỉ dựng lại TrackBundle từ đĩa.
  * Kết quả render cũng được cache: id = hash(reference, target, tham số, mã nguồn engine), nên
    bấm render lại với đúng tham số cũ trả về ngay, sửa engine/*.py sẽ tự vô hiệu cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from urllib.parse import quote

from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from engine import (
    ANALYSIS_VERSION,
    STAGE_ANALYSIS,
    STAGE_DECODE,
    STAGE_DEMUCS,
    STAGE_MASTER,
    STAGE_MATCH,
    STEM_NAMES,
    RemixParams,
    TrackAnalysis,
    TrackBundle,
    build_plan,
    prepare_track,
    render_remix,
    rubberband_available,
)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("airemix")

# --------------------------------------------------------------------------- #
# Cấu hình
# --------------------------------------------------------------------------- #
APP_VERSION = "1.0"
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CACHE_DIR = BASE_DIR / "cache"                 # engine: cache/<sha1>/{mix.wav, stems/, analysis.json}
UPLOAD_DIR = BASE_DIR / "work" / "uploads"     # file upload tạm, xoá ngay sau khi phân tích xong
OUTPUT_DIR = BASE_DIR / "outputs"              # outputs/<render_id>.wav (+ .json)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


MAX_UPLOAD_MB = _env_int("AIREMIX_MAX_UPLOAD_MB", 300)    # 0 = không giới hạn
MAX_MINUTES = _env_int("AIREMIX_MAX_MINUTES", 12)         # bảo vệ RAM khi tách stem; 0 = tắt
MAX_CACHE_TRACKS = _env_int("AIREMIX_CACHE_TRACKS", 20)   # số bài giữ lại trong cache/; 0 = không xoá
MAX_RENDERS_KEPT = _env_int("AIREMIX_KEEP_RENDERS", 8)    # số bản remix giữ lại trong outputs/
HOST = os.environ.get("AIREMIX_HOST", "127.0.0.1")
PORT = _env_int("AIREMIX_PORT", 8000)

ROLES = {"reference": "Reference", "target": "Target"}
ALLOWED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".wma"}
HASH_RE = re.compile(r"^[0-9a-f]{40}$")
RENDER_ID_RE = re.compile(r"^[0-9a-f]{20}$")
JOB_TTL_SECONDS = 2 * 3600

STAGE_LABELS = {
    STAGE_DECODE: "Giải mã",
    STAGE_DEMUCS: "Tách stem (Demucs)",
    STAGE_ANALYSIS: "Phân tích",
    STAGE_MATCH: "Khớp Vocal & Beat",
    STAGE_MASTER: "Mastering",
}
# (stage, trọng số trong thanh tiến độ tổng của job)
ANALYZE_STAGES = ((STAGE_DECODE, 0.05), (STAGE_DEMUCS, 0.75), (STAGE_ANALYSIS, 0.20))
RENDER_STAGES = ((STAGE_MATCH, 0.60), (STAGE_MASTER, 0.40))
RENDER_STAGES_NO_MASTER = ((STAGE_MATCH, 1.0),)

for _d in (STATIC_DIR, CACHE_DIR, UPLOAD_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class TrackNotReady(LookupError):
    """Bài chưa được phân tích (hoặc cache đã bị xoá / cũ)."""


# --------------------------------------------------------------------------- #
# Job: trạng thái tiến độ an toàn đa luồng
# --------------------------------------------------------------------------- #
_JOBS_LOCK = threading.RLock()
_HEAVY = threading.Lock()                 # chỉ 1 job nặng chạy tại một thời điểm
_STATE_LOCK = threading.Lock()
_CURRENT: dict[str, str] = {}             # role -> hash bài đang dùng (được bảo vệ khỏi dọn cache)
JOBS: dict[str, "Job"] = {}


class Job:
    def __init__(self, kind: str, stages: tuple, role: Optional[str] = None, pins: tuple = ()):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.role = role
        self.stages = [s for s, _ in stages]
        total = sum(w for _, w in stages)
        self.weights = {s: w / total for s, w in stages}
        self.fractions = {s: 0.0 for s in self.stages}
        self.pins = frozenset(pins)       # hash cần giữ nguyên trong cache khi job chưa xong
        self.status = "queued"            # queued | running | done | error
        self.stage = "queued"
        self.message = "Đang chờ tới lượt xử lý..."
        self.progress = 0.0
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None

    # progress_cb(stage, fraction, message) của engine
    def update(self, stage: str, fraction: float, message: str) -> None:
        frac = float(fraction) if math.isfinite(float(fraction)) else 0.0
        frac = min(1.0, max(0.0, frac))
        with _JOBS_LOCK:
            if self.status in ("done", "error"):
                return
            self.status = "running"
            if stage in self.fractions:
                idx = self.stages.index(stage)
                current = self.stages.index(self.stage) if self.stage in self.fractions else -1
                if idx >= current:           # bỏ qua cập nhật muộn của bước đã qua: không cho lùi
                    for earlier in self.stages[:idx]:
                        self.fractions[earlier] = 1.0
                    self.fractions[stage] = max(self.fractions[stage], frac)
                    self.stage = stage
            self.message = str(message)
            self.progress = sum(self.weights[s] * self.fractions[s] for s in self.stages)

    def start(self) -> None:
        with _JOBS_LOCK:
            self.status = "running"
            self.stage = self.stages[0]
            self.message = "Bắt đầu xử lý..."
            self.started = time.time()

    def finish(self, result: dict) -> None:
        with _JOBS_LOCK:
            self.fractions = {s: 1.0 for s in self.stages}
            self.progress = 1.0
            self.status = "done"
            self.result = result
            self.message = "Hoàn tất"
            self.finished = time.time()

    def fail(self, error: str) -> None:
        with _JOBS_LOCK:
            self.status = "error"
            self.error = error
            self.message = error
            self.finished = time.time()

    def to_dict(self) -> dict:
        with _JOBS_LOCK:
            steps = []
            for s in self.stages:
                frac = self.fractions[s]
                if self.status == "done" or frac >= 1.0:
                    state = "done"
                elif s == self.stage and self.status == "running":
                    state = "active"
                elif s == self.stage and self.status == "error":
                    state = "error"
                else:
                    state = "pending"
                steps.append({"id": s, "label": STAGE_LABELS[s], "fraction": round(frac, 4), "state": state})
            started = self.started or self.created
            return {
                "id": self.id,
                "kind": self.kind,
                "role": self.role,
                "status": self.status,
                "stage": self.stage,
                "message": self.message,
                "progress": round(self.progress, 4),
                "steps": steps,
                "result": self.result,
                "error": self.error,
                "elapsed": round((self.finished or time.time()) - started, 1),
            }


def _new_job(kind: str, stages: tuple, role: Optional[str] = None, pins: tuple = ()) -> Job:
    job = Job(kind, stages, role, pins)
    with _JOBS_LOCK:
        now = time.time()
        for jid in [j.id for j in JOBS.values() if j.finished and now - j.finished > JOB_TTL_SECONDS]:
            del JOBS[jid]
        while len(JOBS) >= 200:
            oldest_done = next((j.id for j in JOBS.values() if j.finished), None)
            if oldest_done is None:
                break
            del JOBS[oldest_done]
        JOBS[job.id] = job
    return job


def _describe_error(exc: Exception) -> str:
    name = type(exc).__name__
    if isinstance(exc, (ValueError, TrackNotReady)):
        return str(exc)          # thông báo tiếng Việt từ engine / từ server
    if isinstance(exc, MemoryError) or name == "OutOfMemoryError":
        return ("Hết bộ nhớ khi xử lý. Hãy thử file ngắn hơn hoặc đóng bớt ứng dụng khác "
                "(với GPU: chạy Demucs bằng CPU).")
    if name in {"NoBackendError", "LibsndfileError"}:
        return ("Không giải mã được file âm thanh. Hãy cài FFmpeg (cho mp3/m4a/aac) "
                "hoặc dùng file WAV/FLAC.")
    return f"{name}: {exc}"


def _run_job(job: Job, fn: Callable[..., dict], *args: Any) -> None:
    """Chạy fn(job, *args) trong luồng nền, tuần tự với mọi job nặng khác."""
    try:
        if not _HEAVY.acquire(blocking=False):
            log.info("Job %s xếp hàng chờ job đang chạy", job.id)
            _HEAVY.acquire()
        try:
            job.start()
            result = fn(job, *args)
        finally:
            _HEAVY.release()
        job.finish(result)
    except Exception as exc:  # noqa: BLE001 - mọi lỗi đều phải tới được giao diện
        log.exception("Job %s (%s) thất bại", job.id, job.kind)
        job.fail(_describe_error(exc))


# --------------------------------------------------------------------------- #
# Cache: nạp bundle từ đĩa, dọn dẹp
# --------------------------------------------------------------------------- #
def _touch(path: Path) -> None:
    try:
        os.utime(path, None)
    except OSError:
        pass


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_bundle(file_hash: str) -> TrackBundle:
    """Dựng TrackBundle từ cache/<sha1>/ (không đọc audio vào RAM)."""
    file_hash = (file_hash or "").lower()
    if not HASH_RE.fullmatch(file_hash):
        raise TrackNotReady("Mã bài hát không hợp lệ.")
    root = CACHE_DIR / file_hash
    analysis_path = root / "analysis.json"
    try:
        analysis = TrackAnalysis.from_dict(json.loads(analysis_path.read_text(encoding="utf-8")))
    except (OSError, KeyError, TypeError, ValueError):
        raise TrackNotReady(
            "Chưa có dữ liệu phân tích cho bài này (cache có thể đã bị xoá). Hãy tải file lên lại."
        ) from None
    if analysis.version != ANALYSIS_VERSION:
        raise TrackNotReady("Dữ liệu phân tích của bài này thuộc phiên bản cũ. Hãy tải file lên lại.")
    bundle = TrackBundle(file_hash=file_hash, root=root, analysis=analysis)
    needed = [bundle.mix_path, *(bundle.stem_path(name) for name in STEM_NAMES)]
    if any(not p.exists() for p in needed):
        raise TrackNotReady("Cache của bài này không đầy đủ (thiếu stem). Hãy tải file lên lại.")
    _touch(analysis_path)  # đánh dấu "vừa dùng" cho việc dọn cache theo LRU
    return bundle


def _read_name(root: Path) -> Optional[str]:
    try:
        return json.loads((root / "meta.json").read_text(encoding="utf-8")).get("name")
    except (OSError, ValueError, AttributeError):
        return None


def _pinned_hashes() -> set[str]:
    with _JOBS_LOCK:
        pinned: set[str] = set()
        for job in JOBS.values():
            if job.status in ("queued", "running"):
                pinned |= job.pins
        return pinned


def _prune_cache() -> None:
    """Giữ tối đa MAX_CACHE_TRACKS bài (mới dùng nhất); không xoá bài đang dùng / đang chờ render."""
    if MAX_CACHE_TRACKS <= 0:
        return
    with _STATE_LOCK:
        protected = set(_CURRENT.values())
    protected |= _pinned_hashes()
    entries: list[tuple[float, Path]] = []
    for d in CACHE_DIR.iterdir():
        if not (d.is_dir() and HASH_RE.fullmatch(d.name)):
            continue
        try:
            mtime = d.stat().st_mtime
            ap = d / "analysis.json"
            if ap.exists():
                mtime = max(mtime, ap.stat().st_mtime)
        except OSError:
            continue
        entries.append((mtime, d))
    entries.sort(key=lambda e: e[0], reverse=True)
    for _, d in entries[MAX_CACHE_TRACKS:]:
        if d.name in protected:
            continue
        log.info("Dọn cache: xoá %s", d.name)
        shutil.rmtree(d, ignore_errors=True)


def _prune_outputs(keep_id: Optional[str] = None) -> None:
    if MAX_RENDERS_KEPT <= 0:
        return
    wavs = []
    for p in OUTPUT_DIR.glob("*.wav"):
        if RENDER_ID_RE.fullmatch(p.stem):
            try:
                wavs.append((p.stat().st_mtime, p))
            except OSError:
                continue
    wavs.sort(key=lambda e: e[0], reverse=True)
    for _, p in wavs[MAX_RENDERS_KEPT:]:
        if p.stem == keep_id:
            continue
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)


def _engine_fingerprint() -> str:
    parts = []
    for p in sorted((BASE_DIR / "engine").glob("*.py")):
        st = p.stat()
        parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


_ENGINE_FP = _engine_fingerprint()


# --------------------------------------------------------------------------- #
# Tác vụ nền
# --------------------------------------------------------------------------- #
def _probe_duration(path: Path) -> Optional[float]:
    """Độ dài (giây) đọc từ header file; None nếu không đọc được (để engine tự xử lý)."""
    try:
        import soundfile as sf

        dur = float(sf.info(str(path)).duration)
        if dur > 0:
            return dur
    except Exception:  # noqa: BLE001
        pass
    try:
        import librosa

        dur = float(librosa.get_duration(path=str(path)))
        return dur if dur > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _track_payload(bundle: TrackBundle) -> dict:
    return {
        "hash": bundle.file_hash,
        "name": _read_name(bundle.root) or bundle.file_hash[:8],
        "analysis": bundle.analysis.summary(),
        "audio_url": f"/api/tracks/{bundle.file_hash}/audio",
    }


def _analyze_task(job: Job, upload_path: Path, role: str, display_name: str) -> dict:
    try:
        if MAX_MINUTES > 0:
            duration = _probe_duration(upload_path)
            if duration is not None and duration > MAX_MINUTES * 60:
                raise ValueError(
                    f"Bài dài {duration / 60:.1f} phút, vượt giới hạn {MAX_MINUTES} phút "
                    f"(đặt biến môi trường AIREMIX_MAX_MINUTES để đổi)."
                )
        prepared = prepare_track(upload_path, CACHE_DIR, job.update, label=ROLES[role])
        _write_json_atomic(prepared.root / "meta.json", {"name": display_name})
        bundle = load_bundle(prepared.file_hash)   # nạp lại từ đĩa: dữ liệu JSON thuần, và đánh dấu LRU
        with _STATE_LOCK:
            _CURRENT[role] = bundle.file_hash
        _prune_cache()
        payload = _track_payload(bundle)
        payload["role"] = role
        return payload
    finally:
        upload_path.unlink(missing_ok=True)


def _render_id(ref_hash: str, tgt_hash: str, params: RemixParams) -> str:
    payload = json.dumps(
        {"r": ref_hash, "t": tgt_hash, "p": asdict(params), "e": _ENGINE_FP, "v": ANALYSIS_VERSION},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def _safe_stem(name: Optional[str]) -> str:
    stem = re.sub(r"[^\w\-. ()]+", "_", Path(name or "").stem).strip(" ._")
    return stem[:60] or "track"


def _render_result(info: dict, cached: bool) -> dict:
    rid = info["render_id"]
    return {
        **info,
        "cached": cached,
        "audio_url": f"/api/renders/{rid}/audio",
        "download_url": f"/api/renders/{rid}/download",
    }


def _render_task(job: Job, ref_hash: str, tgt_hash: str, params: RemixParams, render_id: str) -> dict:
    out_wav = OUTPUT_DIR / f"{render_id}.wav"
    meta_path = OUTPUT_DIR / f"{render_id}.json"

    # Cùng tham số + cùng engine -> dùng lại file đã render (kiểm tra lại trong khóa để tránh render đôi)
    if out_wav.exists() and meta_path.exists():
        try:
            info = json.loads(meta_path.read_text(encoding="utf-8"))
            _touch(out_wav)
            job.update(STAGE_MATCH, 1.0, "Dùng lại kết quả đã render trước đó (cache)")
            return _render_result(info, cached=True)
        except (OSError, ValueError):
            pass

    ref = load_bundle(ref_hash)
    tgt = load_bundle(tgt_hash)
    tmp_wav = OUTPUT_DIR / f"{render_id}.tmp.wav"
    started = time.time()
    try:
        report = render_remix(ref, tgt, params, tmp_wav, job.update)
        tmp_wav.replace(out_wav)
    finally:
        tmp_wav.unlink(missing_ok=True)

    ref_name = _read_name(ref.root) or "reference"
    tgt_name = _read_name(tgt.root) or "target"
    info = {
        "render_id": render_id,
        "filename": f"{_safe_stem(tgt_name)} x {_safe_stem(ref_name)} remix.wav",
        "reference": ref_hash,
        "target": tgt_hash,
        "duration": report["duration"],
        "plan": report["plan"],
        "vocal_engine": report["vocal_engine"],
        "vocal_gain_db": report["vocal_gain_db"],
        "mastering_engine": report["mastering_engine"],
        "params": report["params"],
        "render_seconds": round(time.time() - started, 1),
    }
    _write_json_atomic(meta_path, info)
    _prune_outputs(keep_id=render_id)
    return _render_result(info, cached=False)


# --------------------------------------------------------------------------- #
# Phục vụ file (hỗ trợ HTTP Range để Wavesurfer / <audio> tua được)
# --------------------------------------------------------------------------- #
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _parse_range(header: Optional[str], size: int) -> Optional[tuple[int, int]]:
    """Trả về (start, end) gồm cả hai đầu; None = phục vụ cả file; 416 nếu ngoài phạm vi."""
    if not header:
        return None
    match = _RANGE_RE.match(header.strip())
    if not match:
        return None
    first, last = match.groups()
    if first == "" and last == "":
        return None
    if first == "":                       # bytes=-N : N byte cuối
        count = int(last)
        if count == 0 or size == 0:
            raise HTTPException(416, "Range không hợp lệ", headers={"Content-Range": f"bytes */{size}"})
        return max(size - count, 0), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if last and end < start:
        return None
    if start >= size:
        raise HTTPException(416, "Range không hợp lệ", headers={"Content-Range": f"bytes */{size}"})
    return start, min(end, size - 1)


def _iter_file(path: Path, start: int, length: int, chunk: int = 256 * 1024) -> Iterator[bytes]:
    with open(path, "rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            data = handle.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def _content_disposition(filename: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r'[^A-Za-z0-9._ ()-]+', "_", ascii_name).strip() or "remix.wav"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def _file_response(request: Request, path: Path, media_type: str, *,
                   download_name: Optional[str] = None) -> StreamingResponse:
    size = path.stat().st_size
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    if download_name:
        headers["Content-Disposition"] = _content_disposition(download_name)
    byte_range = _parse_range(request.headers.get("range"), size)
    if byte_range is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_iter_file(path, 0, size), media_type=media_type, headers=headers)
    start, end = byte_range
    length = end - start + 1
    headers["Content-Length"] = str(length)
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(_iter_file(path, start, length), status_code=206,
                             media_type=media_type, headers=headers)


# --------------------------------------------------------------------------- #
# Kiểm tra dữ liệu vào
# --------------------------------------------------------------------------- #
def _check_role(role: str) -> str:
    role = (role or "").lower()
    if role not in ROLES:
        raise HTTPException(422, "role phải là 'reference' hoặc 'target'.")
    return role


def _require_hash(value: Any, field: str) -> str:
    text = value.lower() if isinstance(value, str) else ""
    if not HASH_RE.fullmatch(text):
        raise HTTPException(422, f"Thiếu hoặc sai mã '{field}'. Hãy tải và phân tích bài trước.")
    return text


def _as_float(key: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HTTPException(422, f"Giá trị '{key}' phải là số.") from None
    if not math.isfinite(number):
        raise HTTPException(422, f"Giá trị '{key}' không hợp lệ.")
    return number


def _as_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "0", "no", "off"}:
        return False
    raise HTTPException(422, f"Giá trị '{key}' phải là true/false.")


def _parse_params(body: dict) -> RemixParams:
    data: dict[str, Any] = {}
    for key in ("vocal_volume", "beat_volume", "sidechain_depth", "reverb_mix", "anchor_time"):
        value = body.get(key)
        if value is not None and value != "":
            data[key] = _as_float(key, value)
    shift = body.get("semitone_shift")
    if shift is not None and shift != "":
        data["semitone_shift"] = max(-12, min(12, int(round(_as_float("semitone_shift", shift)))))
    if body.get("master") is not None:
        data["master"] = _as_bool("master", body["master"])
    return RemixParams.from_dict(data)   # tự clamp về khoảng hợp lệ


def _clean_name(raw: Optional[str]) -> str:
    name = Path((raw or "").replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f]", "", name).strip()
    return name[:150] or "audio"


def _bundle_or_404(file_hash: str) -> TrackBundle:
    try:
        return load_bundle(file_hash)
    except TrackNotReady as exc:
        raise HTTPException(404, str(exc)) from None


# --------------------------------------------------------------------------- #
# Ứng dụng
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_: FastAPI):
    for stale in UPLOAD_DIR.iterdir():           # file upload dở dang của lần chạy trước
        stale.unlink(missing_ok=True)
    for stale in OUTPUT_DIR.glob("*.tmp*.wav"):
        stale.unlink(missing_ok=True)
    _prune_cache()
    _prune_outputs()
    if not rubberband_available():
        log.warning("Không thấy 'rubberband' trong PATH: co giãn/đổi tông sẽ dùng librosa (chất lượng thấp hơn).")
    if shutil.which("ffmpeg") is None:
        log.warning("Không thấy 'ffmpeg' trong PATH: mp3/m4a/aac có thể không đọc được.")
    log.info("Sẵn sàng tại http://localhost:%d", PORT)
    yield


app = FastAPI(title="AI Remix Style Transfer Studio", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/system")
def system_info() -> dict:
    return {
        "version": APP_VERSION,
        "rubberband": rubberband_available(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "max_minutes": MAX_MINUTES},
        "extensions": sorted(ALLOWED_EXT),
        "stages": {
            "analyze": [{"id": s, "label": STAGE_LABELS[s]} for s, _ in ANALYZE_STAGES],
            "render": [{"id": s, "label": STAGE_LABELS[s]} for s, _ in RENDER_STAGES],
        },
    }


@app.post("/api/analyze/{role}")
def analyze_upload(role: str, background: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    role = _check_role(role)
    name = _clean_name(file.filename)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Định dạng '{ext or 'không rõ'}' chưa được hỗ trợ. "
                                 f"Dùng: {', '.join(sorted(ALLOWED_EXT))}.")
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                block = file.file.read(1 << 20)
                if not block:
                    break
                size += len(block)
                if limit and size > limit:
                    raise HTTPException(413, f"File vượt giới hạn {MAX_UPLOAD_MB} MB.")
                out.write(block)
        if size == 0:
            raise HTTPException(400, "File rỗng.")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    job = _new_job("analyze", ANALYZE_STAGES, role=role)
    background.add_task(_run_job, job, _analyze_task, dest, role, name)
    return {"job_id": job.id, "role": role, "name": name}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy job (đã hết hạn hoặc server đã khởi động lại).")
    return job.to_dict()


@app.get("/api/tracks/{file_hash}")
def track_info(file_hash: str) -> dict:
    return _track_payload(_bundle_or_404(file_hash))


@app.get("/api/tracks/{file_hash}/audio")
def track_audio(file_hash: str, request: Request):
    bundle = _bundle_or_404(file_hash)
    return _file_response(request, bundle.mix_path, "audio/wav")


@app.post("/api/render")
def render(background: BackgroundTasks, body: dict[str, Any] = Body(...)) -> dict:
    ref_hash = _require_hash(body.get("reference"), "reference")
    tgt_hash = _require_hash(body.get("target"), "target")
    params = _parse_params(body)
    try:
        ref = load_bundle(ref_hash)
        tgt = load_bundle(tgt_hash)
    except TrackNotReady as exc:
        raise HTTPException(404, str(exc)) from None
    try:
        plan = build_plan(ref.analysis, tgt.analysis, params)   # lỗi rõ ràng trước khi xếp hàng
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None

    render_id = _render_id(ref_hash, tgt_hash, params)
    stages = RENDER_STAGES if params.master else RENDER_STAGES_NO_MASTER
    job = _new_job("render", stages, pins=(ref_hash, tgt_hash))
    background.add_task(_run_job, job, _render_task, ref_hash, tgt_hash, params, render_id)
    return {"job_id": job.id, "render_id": render_id, "plan": asdict(plan), "params": asdict(params)}


# Nếu file WAV bị mất nhưng metadata còn, tự dựng lại từ cache stem đã có.
_RENDER_REPAIR_LOCK = threading.Lock()

def _render_paths(render_id: str) -> tuple[Path, Path]:
    if not RENDER_ID_RE.fullmatch(render_id or ""):
        raise HTTPException(404, "Mã bản remix không hợp lệ.")
    wav, meta = OUTPUT_DIR / f"{render_id}.wav", OUTPUT_DIR / f"{render_id}.json"
    if not meta.is_file():
        raise HTTPException(404, "Không tìm thấy thông tin bản remix. Hãy render lại.")
    if not wav.is_file() or wav.stat().st_size <= 44:
        with _RENDER_REPAIR_LOCK:
            # Kiểm tra lại sau khi lấy khóa, phòng hai request cùng sửa một file.
            if not wav.is_file() or wav.stat().st_size <= 44:
                try:
                    info = json.loads(meta.read_text(encoding="utf-8"))
                    ref = load_bundle(info["reference"])
                    tgt = load_bundle(info["target"])
                    params = RemixParams.from_dict(info.get("params") or {})
                    tmp = OUTPUT_DIR / f"{render_id}.repair.tmp.wav"
                    try:
                        render_remix(ref, tgt, params, tmp)
                        if not tmp.is_file() or tmp.stat().st_size <= 44:
                            raise RuntimeError("Bản WAV dựng lại bị rỗng.")
                        tmp.replace(wav)
                        log.warning("Đã tự khôi phục file WAV cho render %s", render_id)
                    finally:
                        tmp.unlink(missing_ok=True)
                except Exception as exc:
                    log.exception("Không thể tự khôi phục render %s", render_id)
                    raise HTTPException(
                        404,
                        "File WAV của bản remix đã mất và không thể tự khôi phục. "
                        "Hãy tải lại hai bài và render lại. Chi tiết: " + str(exc)
                    ) from None
    return wav, meta


@app.get("/api/renders/{render_id}")
def render_info(render_id: str) -> dict:
    _, meta = _render_paths(render_id)
    try:
        return _render_result(json.loads(meta.read_text(encoding="utf-8")), cached=True)
    except (OSError, ValueError):
        raise HTTPException(404, "Thông tin bản remix bị hỏng. Hãy render lại.") from None


@app.get("/api/renders/{render_id}/audio")
def render_audio(render_id: str, request: Request):
    wav, _ = _render_paths(render_id)
    return _file_response(request, wav, "audio/wav")


@app.get("/api/renders/{render_id}/download")
def render_download(render_id: str, request: Request):
    wav, meta = _render_paths(render_id)
    try:
        filename = json.loads(meta.read_text(encoding="utf-8")).get("filename") or f"remix-{render_id}.wav"
    except (OSError, ValueError):
        filename = f"remix-{render_id}.wav"
    return _file_response(request, wav, "audio/wav", download_name=filename)


_PLACEHOLDER = """<!doctype html><meta charset="utf-8"><title>AI Remix Style Transfer Studio</title>
<body style="font-family:system-ui;background:#0b0d10;color:#d7dae0;padding:3rem">
<h2>AI Remix Style Transfer Studio</h2>
<p>API đã chạy. Chưa có <code>static/index.html</code> (giao diện là Phần 3).</p>
<p>Tài liệu API: <a style="color:#7cc4ff" href="/docs">/docs</a></p></body>"""


@app.get("/", include_in_schema=False)
def index():
    page = STATIC_DIR / "index.html"
    if page.is_file():
        return FileResponse(page, headers={"Cache-Control": "no-cache"})
    return HTMLResponse(_PLACEHOLDER)


# Đăng ký sau cùng để không che các route /api/*
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
