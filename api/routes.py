"""API 路由定义。"""
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from api.monitor_utils import build_agent_event_funnel
from core.knowledge_graph import build_sample_math_graph
from core.observability import get_http_metrics_snapshot

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


@router.get("/monitor/summary")
async def monitor_summary(
    request: Request,
    learner_id: Optional[str] = Query(None, description="用于掌握度分布的学习者 ID"),
) -> dict[str, Any]:
    """
    竞赛可展示监控汇总：HTTP 延迟/错误率、LLM 指标、掌握度分布、Agent 事件漏斗。
    """
    orch = request.app.state.orchestrator
    metrics = get_http_metrics_snapshot()
    bus_stats = orch.get_event_bus_stats()
    by_type = bus_stats.get("by_type") or {}
    funnel = build_agent_event_funnel(by_type if isinstance(by_type, dict) else {})

    mastery_payload: dict[str, Any] = {
        "learner_id": learner_id,
        "buckets": None,
        "per_knowledge": None,
    }
    if learner_id and learner_id.strip():
        lid = learner_id.strip()
        progress = orch.get_learner_progress(lid)
        kg = build_sample_math_graph()
        if progress.get("status") != "no_data":
            model = orch.learner_model_manager.get_or_create_model(lid)
            per_knowledge = []
            for kid, node in kg.nodes.items():
                st = model.get_state(kid)
                per_knowledge.append(
                    {
                        "id": kid,
                        "name": node.name,
                        "mastery": round(float(st.mastery), 4),
                        "attempts": int(st.attempts),
                    }
                )
            per_knowledge.sort(key=lambda x: x["mastery"])

            def _bucket(m: float) -> str:
                if m >= 0.85:
                    return "精通(≥85%)"
                if m >= 0.6:
                    return "熟练(60–85%)"
                if m >= 0.3:
                    return "发展中(30–60%)"
                return "待加强(<30%)"

            buckets: dict[str, int] = {}
            for row in per_knowledge:
                b = _bucket(row["mastery"])
                buckets[b] = buckets.get(b, 0) + 1
            mastery_payload["buckets"] = buckets
            mastery_payload["per_knowledge"] = per_knowledge
        else:
            mastery_payload["note"] = "该学习者暂无模型数据，请先答题或提问。"

    return {
        "generated_at": datetime.now().isoformat(),
        "metrics": metrics,
        "event_bus": {
            "total_published": bus_stats.get("total_published"),
            "total_handled": bus_stats.get("total_handled"),
            "total_in_history": bus_stats.get("total_in_history"),
            "dead_letter_count": bus_stats.get("dead_letter_count"),
            "active_subscriptions": bus_stats.get("active_subscriptions"),
        },
        "agent_funnel": funnel,
        "mastery": mastery_payload,
    }