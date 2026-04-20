"""
FastAPI 应用入口。

启动方式：
    cd python/
    python -m api.main
    或
    uvicorn api.main:app --reload --port 8000
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.websocket import ws_router
from api.orchestrator import AgentOrchestrator
from core.observability import record_http_request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)

orchestrator: AgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = AgentOrchestrator()
    app.state.orchestrator = orchestrator
    logging.getLogger(__name__).info("Agent orchestrator started with 5 agents")
    yield
    try:
        app.state.orchestrator.db.close()
    except Exception:
        logging.getLogger(__name__).exception("Failed to close orchestrator resources")
    logging.getLogger(__name__).info("Shutting down")


app = FastAPI(
    title="多Agent智能教育系统",
    description=(
        "5-Agent Mesh+事件驱动架构的个性化学习系统。\n\n"
        "**Agent列表：**\n"
        "- Assessment Agent：知识点评估\n"
        "- Tutor Agent：苏格拉底式教学\n"
        "- Curriculum Agent：学习路径规划\n"
        "- Hint Agent：分级提示\n"
        "- Engagement Agent：互动监测"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """记录 API 延迟与状态码（跳过文档等静态端点）。"""
    path = request.url.path
    if path in ("/docs", "/redoc", "/openapi.json", "/favicon.ico"):
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if path.startswith("/api/"):
        record_http_request(path, response.status_code, latency_ms)
    return response


app.include_router(router, prefix="/api/v1")
app.include_router(ws_router)

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
