# AI Remix Style Transfer Studio — tài liệu bàn giao

Web app chạy local (Python 3.10+, FastAPI). Người dùng tải lên **Reference Remix** và **Target Song**.
Hệ thống tách stem bằng Demucs (htdemucs), phân tích BPM/Key/Drop/Sidechain/phổ tần của Reference,
co giãn + đổi tông vocal của Target, đặt vào Drop của Reference, thêm Reverb/Delay + sidechain theo
kick của Reference, rồi master bằng Matchering.

Ngôn ngữ làm việc với chủ dự án: **tiếng Việt** (thông báo UI, lỗi, comment đều tiếng Việt).

## Trạng thái

| Phần | Nội dung | Trạng thái |
|---|---|---|
| 1 | `engine/analyzer.py`, `engine/remixer.py` | Xong, **không sửa nếu không có lý do** (chủ dự án đã duyệt) |
| 2 | `main.py` (FastAPI + Uvicorn) | Xong, đã qua test logic, đang chờ chủ dự án xác nhận |
| 3 | `static/index.html` | Xong bản đầu, đã thử với server giả lập (43 kiểm tra); **chờ chủ dự án xác nhận + chạy thử end-to-end với engine thật** |

## Quy tắc chủ dự án đặt ra

- Code phải **chạy được thật**: không `# TODO`, không dữ liệu giả lập/mock trong sản phẩm.
- Tối ưu bộ nhớ bằng cache theo hash file (engine đã làm: `cache/<sha1>/`).
- Làm xong từng phần thì **dừng để chủ dự án xác nhận**, không tự làm sang phần kế.
- Nói rõ cái gì đã kiểm chứng và cái gì chưa (sandbox của Claude không có mạng/Demucs).

## Chạy & test

```bash
pip install -r engine/requirements.txt     # cần FFmpeg trong PATH; rubberband-cli là tùy chọn
python main.py                              # http://localhost:8000  (bind 127.0.0.1)
python tests/test_main_logic.py             # 71 kiểm tra logic của main.py (stub fastapi/librosa)
```

Test chỉ phủ logic của `main.py` (Range, tham số, job, hàng đợi, cache, upload, render cache).
**Chưa từng chạy** server thật qua HTTP, Demucs thật hay Matchering thật — hãy chạy thử khi có môi trường.
Biến môi trường: `AIREMIX_HOST`, `AIREMIX_PORT`, `AIREMIX_MAX_UPLOAD_MB`(300), `AIREMIX_MAX_MINUTES`(12),
`AIREMIX_CACHE_TRACKS`(20), `AIREMIX_KEEP_RENDERS`(8).

## Kiến trúc

```
main.py                 FastAPI: upload, job nền, tiến độ, render, phục vụ audio (Range) + static/
engine/analyzer.py      prepare_track(): giải mã -> Demucs -> phân tích; cache theo SHA-1
engine/remixer.py       render_remix(): khớp tempo/tông, đặt vào Drop, FX, sidechain, Matchering
static/index.html       (Phần 3) giao diện
cache/<sha1>/           mix.wav, stems/*.flac, analysis.json, meta.json (tên file gốc)
outputs/<render_id>.wav (+ .json)   bản remix đã render
```

Hợp đồng engine:
`bundle = prepare_track(path, cache_dir, progress_cb, label="Reference")`,
`render_remix(ref, target, RemixParams.from_dict(body), out_wav, progress_cb)`,
`progress_cb(stage, fraction, message)` với stage ∈ `decode|demucs|analysis|matching|mastering`.
`RemixParams`: `vocal_volume 0..2`, `beat_volume 0..2`, `sidechain_depth 0..1`, `reverb_mix 0..1`,
`anchor_time (giây trong Target | null)`, `semitone_shift (int | null = tự tính)`, `master (bool)`.

### Quyết định thiết kế của main.py (đừng phá vỡ)

- Mọi job nặng chạy **tuần tự** qua một khóa (`_HEAVY`): Matchering dùng logger toàn cục `mg.log`,
  Demucs giữ model dùng chung. Job chờ có `status: "queued"`.
- `/api/render` nhận **hash của hai bài** (không giữ trạng thái "bài hiện tại" ở server) và dựng lại
  bundle từ đĩa → hoạt động sau khi restart / nhiều tab.
- Render cũng cache: `render_id = sha1(ref, target, params, dấu vân tay engine/*.py)[:20]`. Render lại
  đúng tham số → trả ngay, `cached: true`. Sửa engine → cache tự vô hiệu.
- "Render lại nhanh" = gọi lại `POST /api/render` khác tham số: không tách Demucs lại, vocal đã
  co giãn/đổi tông cũng được engine cache (khi tempo/tông/anchor không đổi chỉ còn trộn + master).
  Đặt `master:false` để bỏ Matchering khi muốn nghe thử nhanh lúc kéo thanh trượt.
- Dọn dẹp: upload xóa ngay sau phân tích; cache giữ N bài gần nhất (bảo vệ bài đang dùng/đang chờ);
  outputs giữ M bản.

## API (đã cài đặt trong main.py)

Lỗi HTTP trả `{"detail": "<chuỗi tiếng Việt>"}`; riêng lỗi schema tự động của FastAPI có thể là
`detail` dạng **mảng** → UI phải xử lý cả hai. Mã hay gặp: 404 (bài/bản remix không còn), 413, 415, 422.

- `GET /api/system` → `{version, rubberband, ffmpeg, limits:{max_upload_mb,max_minutes}, extensions[], stages:{analyze:[{id,label}], render:[...]}}`
- `POST /api/analyze/{reference|target}` — multipart, field `file` → `{job_id, role, name}`
- `GET /api/jobs/{job_id}` →
  `{id, kind:"analyze"|"render", role, status:"queued|running|done|error", stage, message, progress(0..1), steps:[{id,label,fraction,state:"pending|active|done|error"}], result, error, elapsed}`
  Job analyze có 3 bước `decode, demucs, analysis`; job render có `matching, mastering`
  (chỉ `matching` nếu `master:false`). Nên poll ~500 ms. 404 nghĩa là job hết hạn / server restart.
- Kết quả job **analyze** (`result`): `{role, hash, name, audio_url, analysis}` với `analysis` là
  `TrackAnalysis.summary()`: `duration, bpm, key, key_confidence, drop_time, drop_label, drop_score_db,
  drops[{time,label,score_db}], sections, hook_time, hook_label, has_vocals, lufs, peak_db,
  sidechain_depth_db, sidechain_detected, spectral_centroid, spectrum{freqs[],db[]}`.
- `GET /api/tracks/{hash}` → `{hash, name, analysis, audio_url}` (nạp lại UI sau khi refresh);
  `GET /api/tracks/{hash}/audio` (WAV 24-bit, hỗ trợ Range).
- `POST /api/render` JSON `{reference:<hash>, target:<hash>, vocal_volume, beat_volume,
  sidechain_depth, reverb_mix, anchor_time, semitone_shift, master}` (các tham số đều tùy chọn) →
  `{job_id, render_id, plan, params}`. Lỗi 422 tức thì nếu Target không có vocal hoặc Drop quá sát cuối bài.
  `plan`: `{tempo_multiplier, target_bpm, ref_bpm, stretch_factor, semitones, anchor_src, drop_time, src_start, src_end, place_time}`.
- Kết quả job **render** (`result`): `{render_id, filename, reference, target, duration, plan,
  vocal_engine, vocal_gain_db, mastering_engine, params, render_seconds, cached, audio_url, download_url}`.
- `GET /api/renders/{id}` (cùng JSON trên), `/audio` (phát), `/download` (attachment `.wav`).

## Phần 3 — `static/index.html` (đã làm, xem yêu cầu gốc bên dưới)

Một file duy nhất, phục vụ tại `/` (đã có route; `static/` được mount ở `/`).

Yêu cầu từ chủ dự án:
- **Tailwind CDN** + **Wavesurfer.js v7**, **Dark Mode kiểu DAW**.
- **2 khung kéo-thả** (Reference Remix, Target Song) — hiển thị **BPM / Key / Drop** sau khi phân tích.
- **Thanh tiến trình theo bước** (decode → demucs → analysis → matching → mastering), dùng `steps[]` và `message`.
- **Waveform kết quả** (Wavesurfer đọc `audio_url`).
- **Bảng chỉnh** Vocal Volume / Beat Volume / Sidechain Depth / Reverb Mix (khoảng giá trị như `RemixParams`).
- **Nút tải WAV** (`download_url`).

Gợi ý triển khai / điểm cần chú ý:
- Ghim phiên bản CDN cụ thể (không dùng `@latest`); Wavesurfer v7 là ESM (`import WaveSurfer from ...`).
- Chỉ bật nút Render khi cả hai bài đã phân tích xong; hiển thị lỗi `error`/`detail` rõ ràng
  (ví dụ Target không có vocal). Cảnh báo nếu `/api/system` báo `rubberband:false` (chất lượng đổi tông thấp hơn).
- Khi kéo thanh trượt: debounce rồi gọi lại `POST /api/render`; cân nhắc công tắc "Nghe nhanh (không master)" = `master:false`.
- Lưu hash hai bài (localStorage) để refresh trang nạp lại bằng `GET /api/tracks/{hash}`; nếu 404 thì reset khung đó.
- Hủy/ghi đè polling cũ khi có job mới; dừng poll khi `status` là `done`/`error`.
- Không dùng dữ liệu giả; mọi số liệu lấy từ API. Phổ tần (`analysis.spectrum`) có thể vẽ canvas nhỏ trong khung bài.

Sau khi xong Phần 3: chạy thử end-to-end trên máy thật (2 file nhạc ngắn), báo kết quả, rồi **dừng chờ xác nhận**.
