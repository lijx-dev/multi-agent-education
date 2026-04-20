"""API 路由定义。"""
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Any

router = APIRouter()
logger = logging.getLogger(__name__)


class SubmissionRequest(BaseModel):
    learner_id: str
    knowledge_id: str
    is_correct: bool
    time_spent_seconds: float = 0.0


class QuestionRequest(BaseModel):
    learner_id: str
    knowledge_id: str
    question: str


class MessageRequest(BaseModel):
    learner_id: str
    message: str
    knowledge_id: str = "general"


def _validate_required_fields(learner_id: str, knowledge_id: str) -> None:
    if not learner_id.strip():
        raise ValueError("learner_id 不能为空")
    if not knowledge_id.strip():
        raise ValueError("knowledge_id 不能为空")


@router.post("/submit")
async def submit_answer(req: SubmissionRequest, request: Request) -> dict[str, Any]:
    """
    学生提交答题结果。
    """
    try:
        _validate_required_fields(req.learner_id, req.knowledge_id)
        events = await request.app.state.orchestrator.submit_answer(
            req.learner_id,
            req.knowledge_id,
            req.is_correct,
            req.time_spent_seconds,
        )
        return {
            "learner_id": req.learner_id,
            "events": events,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("submit failed learner=%s knowledge=%s", req.learner_id, req.knowledge_id)
        raise HTTPException(status_code=500, detail="内部服务异常")


@router.post("/question")
async def ask_question(req: QuestionRequest, request: Request) -> dict[str, Any]:
    """
    学生提问。
    """
    try:
        _validate_required_fields(req.learner_id, req.knowledge_id)
        if not req.question.strip():
            raise ValueError("question 不能为空")
        events = await request.app.state.orchestrator.ask_question(
            req.learner_id,
            req.knowledge_id,
            req.question,
        )
        return {
            "learner_id": req.learner_id,
            "events": events,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("question failed learner=%s knowledge=%s", req.learner_id, req.knowledge_id)
        raise HTTPException(status_code=500, detail="内部服务异常")


@router.post("/message")
async def send_message(req: MessageRequest, request: Request) -> dict[str, Any]:
    """
    学生发送自由消息。
    """
    try:
        if not req.learner_id.strip():
            raise ValueError("learner_id 不能为空")
        if not req.message.strip():
            raise ValueError("message 不能为空")
        events = await request.app.state.orchestrator.send_message(
            req.learner_id,
            req.message,
            req.knowledge_id,
        )
        return {
            "learner_id": req.learner_id,
            "events": events,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("message failed learner=%s knowledge=%s", req.learner_id, req.knowledge_id)
        raise HTTPException(status_code=500, detail="内部服务异常")


@router.get("/progress/{learner_id}")
async def get_progress(learner_id: str, request: Request) -> dict[str, Any]:
    """
    获取学习者进度。
    """
    try:
        if not learner_id.strip():
            raise ValueError("learner_id 不能为空")
        return request.app.state.orchestrator.get_learner_progress(learner_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("progress failed learner=%s", learner_id)
        raise HTTPException(status_code=500, detail="内部服务异常")


@router.get("/health")
async def health_check() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}