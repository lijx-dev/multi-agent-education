"""API 路由定义。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any

from api.orchestrator import orchestrator

router = APIRouter()


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


@router.post("/submit")
async def submit_answer(req: SubmissionRequest) -> dict[str, Any]:
    """
    学生提交答题结果。
    """
    try:
        events = await orchestrator.submit_answer(
            req.learner_id,
            req.knowledge_id,
            req.is_correct,
            req.time_spent_seconds,
        )
        return {
            "learner_id": req.learner_id,
            "events": events,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/question")
async def ask_question(req: QuestionRequest) -> dict[str, Any]:
    """
    学生提问。
    """
    try:
        events = await orchestrator.ask_question(
            req.learner_id,
            req.knowledge_id,
            req.question,
        )
        return {
            "learner_id": req.learner_id,
            "events": events,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message")
async def send_message(req: MessageRequest) -> dict[str, Any]:
    """
    学生发送自由消息。
    """
    try:
        events = await orchestrator.send_message(
            req.learner_id,
            req.message,
            req.knowledge_id,
        )
        return {
            "learner_id": req.learner_id,
            "events": events,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{learner_id}")
async def get_progress(learner_id: str) -> dict[str, Any]:
    """
    获取学习者进度。
    """
    try:
        return orchestrator.get_learner_progress(learner_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}