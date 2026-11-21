import io
import json
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# ---- stub các thư viện không có trong sandbox (engine thật vẫn được import) ----
for name in ("librosa", "librosa.effects", "soundfile", "pyloudnorm"):
    sys.modules[name] = mock.MagicMock()


class HTTPException(Exception):
    def __init__(self, status_code, detail=None, headers=None):
        self.status_code, self.detail, self.headers = status_code, detail, headers or {}


class FastAPI:
    def __init__(self, *a, **k): pass
    def _deco(self, *a, **k): return lambda fn: fn
    get = post = _deco
    def mount(self, *a, **k): pass


class StreamingResponse:
    def __init__(self, content, status_code=200, media_type=None, headers=None):
        self.body_iter, self.status_code, self.media_type, self.headers = content, status_code, media_type, headers or {}
    def read(self): return b"".join(self.body_iter)


fa = types.ModuleType("fastapi")
fa.BackgroundTasks = type("BackgroundTasks", (), {
    "__init__": lambda s: setattr(s, "tasks", []),
    "add_task": lambda s, fn, *a: s.tasks.append((fn, a)),
})
fa.Body = lambda *a, **k: None
fa.File = lambda *a, **k: None
fa.FastAPI, fa.HTTPException = FastAPI, HTTPException
fa.Request = type("Request", (), {})
fa.UploadFile = type("UploadFile", (), {})
fr = types.ModuleType("fastapi.responses")
fr.FileResponse = fr.HTMLResponse = type("R", (), {"__init__": lambda s, *a, **k: None})
fr.StreamingResponse = StreamingResponse
fs = types.ModuleType("fastapi.staticfiles")
fs.StaticFiles = lambda *a, **k: None
sys.modules.update({"fastapi": fa, "fastapi.responses": fr, "fastapi.staticfiles": fs})

import main  # noqa: E402
from engine import RemixParams, TrackAnalysis, STEM_NAMES, ANALYSIS_VERSION  # noqa: E402

TMP = Path(tempfile.mkdtemp())
main.CACHE_DIR, main.UPLOAD_DIR, main.OUTPUT_DIR = TMP / "cache", TMP / "up", TMP / "out"
for d in (main.CACHE_DIR, main.UPLOAD_DIR, main.OUTPUT_DIR):
    d.mkdir()

passed = 0


def check(cond, label):
    global passed
    assert cond, f"FAIL: {label}"
    passed += 1
    print(f"  ok  {label}")


def raises(fn, status, label):
    try:
        fn()
    except HTTPException as e:
        assert e.status_code == status, f"FAIL {label}: {e.status_code} != {status}"
        passed_inc(label)
        return e
    raise AssertionError(f"FAIL: {label}: không raise")


def passed_inc(label):
    global passed
    passed += 1
    print(f"  ok  {label}")


def make_analysis(h, bpm=128.0, vocals=True, duration=200.0, drop=40.0):
    period = 60.0 / bpm
    return TrackAnalysis(
        version=ANALYSIS_VERSION, file_hash=h, duration=duration, bpm=bpm, beat_period=period,
        beat_offset=0.1, bar_offset=0.1, key_tonic=9, key_mode="minor", key_name="A minor",
        key_confidence=0.8, drop_time=drop, drop_score_db=12.0,
        drops=[{"time": drop, "beat": 80, "score_db": 12.0}], sections=[], hook_start=30.0,
        has_vocals=vocals, vocal_to_beat_db=-3.0, kick_times=[1.0, 1.5], sidechain_curve=[0.2] * 64,
        sidechain_depth_db=8.0, sidechain_detected=True, lufs=-8.0, peak_db=-0.5,
        spectral_centroid=2000.0, spectrum_freqs=[50.0, 100.0], spectrum_db=[-20.0, -25.0],
    )


def make_cache(h, **kw):
    root = main.CACHE_DIR / h
    (root / "stems").mkdir(parents=True)
    (root / "mix.wav").write_bytes(b"RIFF" + b"\0" * 100)
    for n in STEM_NAMES:
        (root / "stems" / f"{n}.flac").write_bytes(b"x")
    (root / "analysis.json").write_text(json.dumps(make_analysis(h, **kw).to_dict()))
    return root


H_REF, H_TGT, H_BAD = "a" * 40, "b" * 40, "c" * 40

print("[range]")
check(main._parse_range(None, 100) is None, "không header -> full")
check(main._parse_range("bytes=10-19", 100) == (10, 19), "bytes=10-19")
check(main._parse_range("bytes=90-", 100) == (90, 99), "bytes=90-")
check(main._parse_range("bytes=-10", 100) == (90, 99), "suffix")
check(main._parse_range("bytes=50-500", 100) == (50, 99), "end bị cắt về cuối file")
check(main._parse_range("bytes=0-1,5-6", 100) is None, "nhiều range -> full")
check(main._parse_range("bytes=20-10", 100) is None, "range đảo -> full")
raises(lambda: main._parse_range("bytes=100-", 100), 416, "start >= size -> 416")

print("[file response]")
f = TMP / "x.wav"
f.write_bytes(bytes(range(256)) * 4)
req = types.SimpleNamespace(headers={})
r = main._file_response(req, f, "audio/wav")
check(r.status_code == 200 and len(r.read()) == 1024 and r.headers["Content-Length"] == "1024", "200 full")
req = types.SimpleNamespace(headers={"range": "bytes=256-259"})
r = main._file_response(req, f, "audio/wav")
check(r.status_code == 206 and r.read() == bytes([0, 1, 2, 3]) and r.headers["Content-Range"] == "bytes 256-259/1024", "206 đúng byte")
r = main._file_response(types.SimpleNamespace(headers={}), f, "audio/wav", download_name="Bài hát x Ref remix.wav")
cd = r.headers["Content-Disposition"]
check(cd.startswith("attachment;") and "filename*=UTF-8''" in cd and "%C3%A0" in cd or "B%C3%A0i" in cd, "Content-Disposition UTF-8")

print("[params]")
p = main._parse_params({"vocal_volume": "1.5", "beat_volume": 9, "sidechain_depth": -3, "reverb_mix": 0.4,
                        "anchor_time": "12.5", "semitone_shift": 99, "master": "false", "reference": "x"})
check(p.vocal_volume == 1.5 and p.beat_volume == 2.0 and p.sidechain_depth == 0.0, "ép về khoảng hợp lệ")
check(p.anchor_time == 12.5 and p.semitone_shift == 12 and p.master is False, "anchor/semitone/master")
p0 = main._parse_params({})
check(p0 == RemixParams().clamp(), "body rỗng -> mặc định")
raises(lambda: main._parse_params({"vocal_volume": "abc"}), 422, "chuỗi không phải số -> 422")
raises(lambda: main._parse_params({"reverb_mix": float("nan")}), 422, "NaN -> 422")
raises(lambda: main._parse_params({"master": "maybe"}), 422, "master lạ -> 422")
check(main._render_id(H_REF, H_TGT, p) == main._render_id(H_REF, H_TGT, p), "render_id ổn định")
check(main._render_id(H_REF, H_TGT, p) != main._render_id(H_REF, H_TGT, p0), "render_id đổi theo tham số")
check(main._render_id(H_REF, H_TGT, p0) != main._render_id(H_TGT, H_REF, p0), "render_id đổi theo bài")

print("[job]")
j = main.Job("render", main.RENDER_STAGES)
check(j.to_dict()["status"] == "queued", "ban đầu queued")
j.start()
j.update("matching", 0.5, "m")
j.update("matching", 0.2, "m2")  # không được lùi
d = j.to_dict()
check(abs(d["progress"] - 0.3) < 1e-6 and d["steps"][0]["state"] == "active", "tiến độ 0.6*0.5, không lùi")
j.update("mastering", 0.5, "x")
d = j.to_dict()
check(d["steps"][0]["state"] == "done" and d["steps"][1]["state"] == "active", "bước trước tự hoàn tất")
check(abs(d["progress"] - (0.6 + 0.4 * 0.5)) < 1e-6, "tổng tiến độ")
j.update("matching", float("nan"), "nan")
j.fail("lỗi thử")
d = j.to_dict()
check(d["status"] == "error" and d["steps"][1]["state"] == "error", "lỗi đánh dấu đúng bước")
j2 = main.Job("render", main.RENDER_STAGES_NO_MASTER)
j2.start(); j2.update("matching", 1.0, "x"); j2.finish({"a": 1})
check(j2.to_dict()["progress"] == 1.0 and j2.to_dict()["status"] == "done", "no-master hoàn tất")

print("[hàng đợi job]")
order, gate = [], threading.Event()
def slow(job):
    order.append("A-start"); job.update("decode", 0.1, "a"); gate.wait(5); order.append("A-end"); return {"ok": "A"}
def quick(job):
    order.append("B-run"); return {"ok": "B"}
ja, jb = main.Job("analyze", main.ANALYZE_STAGES), main.Job("analyze", main.ANALYZE_STAGES)
ta = threading.Thread(target=main._run_job, args=(ja, slow)); ta.start(); time.sleep(0.2)
tb = threading.Thread(target=main._run_job, args=(jb, quick)); tb.start(); time.sleep(0.3)
check(ja.status == "running" and jb.status == "queued" and order == ["A-start"], "job thứ hai phải chờ")
gate.set(); ta.join(); tb.join()
check(order == ["A-start", "A-end", "B-run"] and ja.status == jb.status == "done", "chạy tuần tự rồi xong")
jc = main.Job("analyze", main.ANALYZE_STAGES)
main._run_job(jc, lambda job: (_ for _ in ()).throw(ValueError("Target Song không có vocal")))
check(jc.status == "error" and jc.error == "Target Song không có vocal", "ValueError giữ nguyên thông báo")
check(main._HEAVY.acquire(blocking=False), "khóa được nhả sau lỗi"); main._HEAVY.release()

print("[load_bundle / cache]")
make_cache(H_REF); make_cache(H_TGT, bpm=100.0)
b = main.load_bundle(H_REF.upper())
check(b.file_hash == H_REF and b.analysis.bpm == 128.0, "nạp bundle từ đĩa (hash viết hoa được chuẩn hoá)")
for bad, label in (("../etc/passwd", "path traversal"), ("zz", "hash sai"), (H_BAD, "không tồn tại")):
    try:
        main.load_bundle(bad); raise AssertionError(label)
    except main.TrackNotReady:
        passed_inc(f"TrackNotReady: {label}")
(main.CACHE_DIR / H_TGT / "stems" / "bass.flac").unlink()
try:
    main.load_bundle(H_TGT); raise AssertionError("thiếu stem")
except main.TrackNotReady as e:
    passed_inc("TrackNotReady: thiếu stem")
(main.CACHE_DIR / H_TGT / "stems" / "bass.flac").write_bytes(b"x")
raises(lambda: main._bundle_or_404(H_BAD), 404, "HTTP 404 cho bài chưa phân tích")

print("[prune cache]")
main.MAX_CACHE_TRACKS = 2
hashes = [f"{i:040x}" for i in range(1, 6)]
for i, h in enumerate(hashes):
    root = make_cache(h)
    os.utime(root / "analysis.json", (1000 + i, 1000 + i))
    os.utime(root, (1000 + i, 1000 + i))
for h in (H_REF, H_TGT):
    os.utime(main.CACHE_DIR / h / "analysis.json", None)
main._CURRENT["reference"] = hashes[0]            # bài cũ nhất nhưng đang được dùng
job = main._new_job("render", main.RENDER_STAGES, pins=(hashes[1],))
main._prune_cache()
left = {p.name for p in main.CACHE_DIR.iterdir()}
check({H_REF, H_TGT} <= left, "2 bài mới dùng nhất được giữ")
check(hashes[0] in left and hashes[1] in left, "bài đang dùng / đang chờ render được bảo vệ")
check(not (left & set(hashes[2:])), "bài cũ khác bị xoá")
job.finish({}); main._CURRENT.clear()
main.MAX_CACHE_TRACKS = 20

print("[upload handler]")
class FakeUpload:
    def __init__(self, name, data): self.filename, self.file = name, io.BytesIO(data)
bg = fa.BackgroundTasks()
out = main.analyze_upload("reference", bg, FakeUpload("Bài hát.mp3", b"abc" * 1000))
check(out["role"] == "reference" and out["name"] == "Bài hát.mp3" and len(bg.tasks) == 1, "upload hợp lệ xếp job")
check(len(list(main.UPLOAD_DIR.iterdir())) == 1, "file tạm đã lưu")
raises(lambda: main.analyze_upload("reference", fa.BackgroundTasks(), FakeUpload("x.exe", b"1")), 415, "đuôi không hỗ trợ -> 415")
raises(lambda: main.analyze_upload("reference", fa.BackgroundTasks(), FakeUpload("x.wav", b"")), 400, "file rỗng -> 400")
raises(lambda: main.analyze_upload("mixer", fa.BackgroundTasks(), FakeUpload("x.wav", b"1")), 422, "role sai -> 422")
main.MAX_UPLOAD_MB = 1
before = len(list(main.UPLOAD_DIR.iterdir()))
raises(lambda: main.analyze_upload("target", fa.BackgroundTasks(), FakeUpload("x.wav", b"0" * (1 << 20) + b"1")), 413, "quá dung lượng -> 413")
check(len(list(main.UPLOAD_DIR.iterdir())) == before, "file dở dang bị xoá khi 413")
main.MAX_UPLOAD_MB = 300

print("[analyze task]")
up = main.UPLOAD_DIR / "t.wav"; up.write_bytes(b"data")
def fake_prepare(path, cache_root, progress, label=""):
    progress("decode", 1.0, f"[{label}] ok"); progress("demucs", 1.0, "d"); progress("analysis", 1.0, "a")
    h = "d" * 40
    root = make_cache(h)
    return types.SimpleNamespace(file_hash=h, root=root)
ja = main.Job("analyze", main.ANALYZE_STAGES, role="target"); ja.start()
with mock.patch.object(main, "prepare_track", fake_prepare), mock.patch.object(main, "_probe_duration", return_value=100.0):
    res = main._analyze_task(ja, up, "target", "Song A.mp3")
check(res["hash"] == "d" * 40 and res["name"] == "Song A.mp3" and res["role"] == "target", "payload phân tích")
check(res["analysis"]["bpm"] == 128.0 and "drop_label" in res["analysis"], "summary gửi cho UI")
check(not up.exists() and main._CURRENT["target"] == "d" * 40, "xoá upload tạm + ghi nhớ bài hiện tại")
up.write_bytes(b"data")
try:
    with mock.patch.object(main, "_probe_duration", return_value=60 * 30.0):
        main._analyze_task(ja, up, "target", "long.mp3")
    raise AssertionError("không chặn bài dài")
except ValueError as e:
    passed_inc("chặn bài quá dài: " + str(e)[:40])
check(not up.exists(), "upload tạm bị xoá cả khi lỗi")

print("[render handler + task]")
bg = fa.BackgroundTasks()
resp = main.render(bg, {"reference": H_REF, "target": H_TGT, "reverb_mix": 0.5, "master": False})
check(resp["plan"]["semitones"] in range(-6, 7) and resp["params"]["reverb_mix"] == 0.5, "trả plan + params")
check(len(bg.tasks) == 1, "render xếp job nền")
raises(lambda: main.render(fa.BackgroundTasks(), {"reference": "nope", "target": H_TGT}), 422, "thiếu hash -> 422")
raises(lambda: main.render(fa.BackgroundTasks(), {"reference": H_REF, "target": H_BAD}), 404, "bài chưa phân tích -> 404")
make_cache("e" * 40, vocals=False)
e = raises(lambda: main.render(fa.BackgroundTasks(), {"reference": H_REF, "target": "e" * 40}), 422, "target không vocal -> 422")
check("vocal" in e.detail, "thông báo lỗi tiếng Việt từ engine")

fn, args = bg.tasks[0]
calls = []
def fake_render(ref, tgt, params, out_path, progress):
    calls.append(params)
    progress("matching", 0.5, "m"); progress("matching", 1.0, "m")
    Path(out_path).write_bytes(b"WAVDATA")
    return {"output": str(out_path), "duration": 200.0, "plan": {"drop_time": 40.0}, "vocal_engine": "librosa",
            "vocal_gain_db": -1.5, "mastering_engine": "none", "params": {"master": False}}
hj = args[0]
check(hj.pins == {H_REF, H_TGT} and hj.status == "queued", "job render ghim 2 bài, đang queued")
with mock.patch.object(main, "render_remix", fake_render):
    fn(*args)                                  # đúng cách BackgroundTasks gọi
d = hj.to_dict()
check(d["status"] == "done" and d["result"]["cached"] is False, "render xong")
rid = d["result"]["render_id"]
check(rid == resp["render_id"], "render_id khớp với phản hồi của POST /api/render")
check((main.OUTPUT_DIR / f"{rid}.wav").read_bytes() == b"WAVDATA" and not list(main.OUTPUT_DIR.glob("*.tmp*")), "ghi nguyên tử, không để file tạm")
check(d["result"]["filename"].endswith("remix.wav") and d["result"]["download_url"].endswith("/download"), "tên file + url tải")
bg2 = fa.BackgroundTasks()
main.render(bg2, {"reference": H_REF, "target": H_TGT, "reverb_mix": 0.5, "master": False})
fn2, args2 = bg2.tasks[0]
with mock.patch.object(main, "render_remix", fake_render):
    fn2(*args2)
check(args2[0].to_dict()["result"]["cached"] is True and len(calls) == 1, "render lại cùng tham số -> dùng cache, không gọi engine")
info = main.render_info(rid)
check(info["audio_url"] == f"/api/renders/{rid}/audio", "GET /api/renders/{id}")
raises(lambda: main.render_info("g" * 20), 404, "render_id sai -> 404")
raises(lambda: main.render_info("0" * 20), 404, "render_id không có -> 404")
dl = main.render_download(rid, types.SimpleNamespace(headers={}))
check("remix.wav" in dl.headers["Content-Disposition"], "download đặt tên đúng")

print("[prune outputs]")
main.MAX_RENDERS_KEPT = 2
for i in range(4):
    stem = f"{i:020x}"
    (main.OUTPUT_DIR / f"{stem}.wav").write_bytes(b"x"); (main.OUTPUT_DIR / f"{stem}.json").write_text("{}")
    os.utime(main.OUTPUT_DIR / f"{stem}.wav", (500 + i, 500 + i))
main._prune_outputs(keep_id=rid)
wavs = {p.stem for p in main.OUTPUT_DIR.glob("*.wav")}
check(rid in wavs and len(wavs) <= 3, f"giữ bản mới + bản vừa render ({len(wavs)} file)")
check(not (main.OUTPUT_DIR / f"{0:020x}.json").exists(), "xoá cả sidecar json")

print("[errors]")
check("FFmpeg" in main._describe_error(type("NoBackendError", (Exception,), {})("x")), "gợi ý FFmpeg")
check("bộ nhớ" in main._describe_error(MemoryError()), "gợi ý hết RAM")

print(f"\nTẤT CẢ {passed} KIỂM TRA ĐẠT")
