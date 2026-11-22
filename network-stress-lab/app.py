from __future__ import annotations

import asyncio
import ipaddress
import os
import secrets
import socket
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ADMIN_TOKEN = os.environ.get("STRESS_LAB_TOKEN", "")
ALLOWED_HOSTS = {
    item.strip().lower().rstrip(".")
    for item in os.environ.get("ALLOWED_TARGET_HOSTS", "").split(",")
    if item.strip()
}
MAX_REQUESTS = 40
MAX_DURATION_SECONDS = 10
MAX_CONCURRENCY = 4
COOLDOWN_SECONDS = 30

app = FastAPI(title="Network Stress Lab", version="1.0.0")
_reports: list[dict] = []
_run_lock = asyncio.Lock()
_last_run_at = 0.0


class RunRequest(BaseModel):
    url: str = Field(min_length=9, max_length=2048)
    requests: int = Field(default=20, ge=1, le=MAX_REQUESTS)
    duration_seconds: int = Field(default=5, ge=1, le=MAX_DURATION_SECONDS)
    concurrency: int = Field(default=2, ge=1, le=MAX_CONCURRENCY)


def require_auth(authorization: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(503, "Chưa cấu hình STRESS_LAB_TOKEN trên máy chủ.")
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not secrets.compare_digest(supplied, ADMIN_TOKEN):
        raise HTTPException(401, "Mã truy cập không hợp lệ.")


async def validate_target(raw_url: str) -> str:
    parsed = urlsplit(raw_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise HTTPException(400, "Chỉ chấp nhận URL HTTPS.")
    if parsed.username or parsed.password or parsed.fragment:
        raise HTTPException(400, "URL chứa thông tin đăng nhập hoặc fragment không được hỗ trợ.")
    host = parsed.hostname.lower().rstrip(".")
    if host not in ALLOWED_HOSTS:
        raise HTTPException(403, "Tên miền không nằm trong ALLOWED_TARGET_HOSTS.")
    if parsed.port not in (None, 443):
        raise HTTPException(400, "Chỉ chấp nhận cổng HTTPS mặc định 443.")
    try:
        ipaddress.ip_address(host)
        raise HTTPException(400, "Không chấp nhận địa chỉ IP trực tiếp.")
    except ValueError:
        pass
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        raise HTTPException(400, "Không thể phân giải DNS của tên miền.")
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise HTTPException(400, "Tên miền không có địa chỉ IP hợp lệ.")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise HTTPException(403, "Tên miền phân giải tới địa chỉ mạng riêng hoặc không công khai.")
    return raw_url


@app.get("/")
async def index():
    return FileResponse(os.path.join(APP_DIR, "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "network-stress-lab", "version": "1.0.0"}


@app.get("/api/config")
async def config(authorization: str | None = Header(default=None)):
    require_auth(authorization)
    return {
        "allowed_hosts": sorted(ALLOWED_HOSTS),
        "max_requests": MAX_REQUESTS,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "max_concurrency": MAX_CONCURRENCY,
        "cooldown_seconds": COOLDOWN_SECONDS,
    }


@app.get("/api/reports")
async def reports(authorization: str | None = Header(default=None)):
    require_auth(authorization)
    return {"reports": _reports[:20]}


@app.post("/api/run")
async def run_test(payload: RunRequest, authorization: str | None = Header(default=None)):
    global _last_run_at
    require_auth(authorization)
    target = await validate_target(payload.url)

    if _run_lock.locked():
        raise HTTPException(429, "Một bài kiểm thử khác đang chạy.")

    async with _run_lock:
        now = time.monotonic()
        remaining = COOLDOWN_SECONDS - (now - _last_run_at)
        if remaining > 0:
            raise HTTPException(429, f"Vui lòng chờ {int(remaining) + 1} giây trước lần kiểm thử tiếp theo.")
        _last_run_at = time.monotonic()
        started = time.monotonic()
        deadline = started + payload.duration_seconds
        semaphore = asyncio.Semaphore(payload.concurrency)
        counter_lock = asyncio.Lock()
        counts = Counter()
        latencies: list[float] = []
        request_budget = payload.requests

        async def worker():
            while time.monotonic() < deadline:
                async with counter_lock:
                    if counts["total"] >= request_budget:
                        return
                    counts["total"] += 1
                async with semaphore:
                    try:
                        before = time.perf_counter()
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(4.0, connect=2.0),
                            follow_redirects=False,
                            trust_env=False,
                            limits=httpx.Limits(max_connections=MAX_CONCURRENCY, max_keepalive_connections=MAX_CONCURRENCY),
                        ) as client:
                            response = await client.get(target, headers={"User-Agent": "NetworkStressLab/1.0 (authorized-load-test)"})
                        elapsed_ms = (time.perf_counter() - before) * 1000
                        latencies.append(elapsed_ms)
                        counts[str(response.status_code)] += 1
                        if 200 <= response.status_code < 400:
                            counts["success"] += 1
                        else:
                            counts["errors"] += 1
                    except (httpx.HTTPError, OSError):
                        counts["errors"] += 1
        await asyncio.gather(*(worker() for _ in range(payload.concurrency)))

        elapsed = max(time.monotonic() - started, 0.001)
        sorted_latencies = sorted(latencies)
        p95 = sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))] if sorted_latencies else 0
        report = {
            "id": uuid.uuid4().hex[:12],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "total_requests": counts["total"],
            "successful": counts["success"],
            "errors": counts["errors"],
            "rps": round(counts["total"] / elapsed, 2),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p95_latency_ms": round(p95, 2),
            "duration_seconds": round(elapsed, 2),
            "status_codes": {key: value for key, value in counts.items() if key.isdigit()},
            "limits": {"requests": request_budget, "duration_seconds": payload.duration_seconds, "concurrency": payload.concurrency},
        }
        _reports.insert(0, report)
        del _reports[20:]
        return report
