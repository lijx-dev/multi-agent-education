"""
WebSocket 实时通信端点。

支持前端与Agent系统的双向实时通信：
- 学生发送消息/答题 → Agent处理 → 实时推送结果

面试要点：
- WebSocket vs HTTP轮询：全双工、低延迟、持久连接
- 断线重连策略：指数退避 + 心跳检测
"""

import json
import logging
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket连接管理器。"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, learner_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections[learner_id] = websocket
        logger.info("WebSocket connected: %s", learner_id)

    async def disconnect(self, learner_id: str):
        async with self._lock:
            self.active_connections.pop(learner_id, None)
        logger.info("WebSocket disconnected: %s", learner_id)

    async def send_to_learner(self, learner_id: str, data: dict):
        async with self._lock:
            ws = self.active_connections.get(learner_id)
        if ws:
            await ws.send_json(data)


manager = ConnectionManager()
MAX_PUSH_EVENTS = 50


@ws_router.websocket("/ws/{learner_id}")
async def websocket_endpoint(websocket: WebSocket, learner_id: str):
    await manager.connect(learner_id, websocket)
    pending_task: asyncio.Task | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_to_learner(
                    learner_id,
                    {"error": "非法JSON格式", "raw": raw[:200]},
                )
                continue
            action = data.get("action", "")

            orch = websocket.app.state.orchestrator

            if action == "submit":
                pending_task = asyncio.create_task(
                    orch.submit_answer(
                        learner_id,
                        data.get("knowledge_id", ""),
                        data.get("is_correct", False),
                        data.get("time_spent_seconds", 0),
                        question_text=data.get("question_text", ""),
                        answer_text=data.get("answer_text", ""),
                        error_type=data.get("error_type"),
                    )
                )
            elif action == "question":
                pending_task = asyncio.create_task(
                    orch.ask_question(
                    learner_id,
                    data.get("knowledge_id", ""),
                    data.get("question", ""),
                    )
                )
            elif action == "message":
                pending_task = asyncio.create_task(
                    orch.send_message(
                    learner_id,
                    data.get("message", ""),
                    data.get("knowledge_id", "general"),
                    )
                )
            else:
                await manager.send_to_learner(learner_id, {"error": f"Unknown action: {action}"})
                continue
            events = await pending_task
            pending_task = None

            for event in events[-MAX_PUSH_EVENTS:]:
                await manager.send_to_learner(
                    learner_id,
                    {
                        "event_type": event.get("type"),
                        "source": event.get("source"),
                        "data": event.get("data", {}),
                    },
                )

    except WebSocketDisconnect:
        if pending_task and not pending_task.done():
            pending_task.cancel()
        await manager.disconnect(learner_id)
    except Exception:
        logger.exception("WebSocket error for learner=%s", learner_id)
        if pending_task and not pending_task.done():
            pending_task.cancel()
        await manager.disconnect(learner_id)
