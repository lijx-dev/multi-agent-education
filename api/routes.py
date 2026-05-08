"""API 路由定义。"""
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request, File, UploadFile
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
    error_type: str | None = None  # 答错时可选：careless / concept / unknown


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
            error_type=req.error_type,
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


@router.get("/review-plan/{learner_id}")
async def get_review_plan(learner_id: str, request: Request) -> dict[str, Any]:
    """SM-2 复习计划：到期项、未来 7 天日程、即将复习知识点。"""
    try:
        if not learner_id.strip():
            raise ValueError("learner_id 不能为空")
        return request.app.state.orchestrator.get_review_plan(learner_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("review-plan failed learner=%s", learner_id)
        raise HTTPException(status_code=500, detail="内部服务异常")


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


# ==================== 错题本相关路由（新增）====================

class WrongQuestionUploadRequest(BaseModel):
    learner_id: str
    knowledge_id: Optional[str] = None
    user_answer: Optional[str] = None
    error_type: str = "unknown"


class WrongQuestionPracticeRequest(BaseModel):
    learner_id: str
    user_answer: str
    is_correct: bool
    time_spent: Optional[int] = None


@router.post("/wrong-question/upload")
async def upload_wrong_question(
    req: WrongQuestionUploadRequest,
    request: Request,
    file: Optional[UploadFile] = File(None)
) -> dict[str, Any]:
    """
    上传错题图片，识别题目并保存到错题本。
    
    Args:
        learner_id: 学习者ID
        knowledge_id: 知识点ID（可选，不提供则自动分析）
        user_answer: 用户答案（可选）
        error_type: 错误类型（concept/careless/unknown）
        file: 图片文件（可选，与image_base64二选一）
    
    Returns:
        处理结果，包含题目信息、解析、生成的练习题
    """
    try:
        if not req.learner_id.strip():
            raise ValueError("learner_id 不能为空")

        # 处理图片数据
        image_path = None
        image_base64 = None
        
        if file:
            # 保存上传的图片
            import os
            from pathlib import Path
            
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            image_path = str(upload_dir / f"{req.learner_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            
            with open(image_path, "wb") as f:
                f.write(await file.read())

        result = request.app.state.orchestrator.upload_wrong_question(
            learner_id=req.learner_id,
            image_path=image_path,
            image_base64=image_base64,
            knowledge_id=req.knowledge_id,
            user_answer=req.user_answer,
            error_type=req.error_type
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("upload wrong question failed learner=%s", req.learner_id)
        raise HTTPException(status_code=500, detail=f"上传错题失败：{str(e)}")


@router.post("/wrong-question/upload-base64")
async def upload_wrong_question_base64(
    learner_id: str,
    image_base64: str,
    knowledge_id: Optional[str] = None,
    user_answer: Optional[str] = None,
    error_type: str = "unknown",
    request: Request = None
) -> dict[str, Any]:
    """
    通过Base64编码上传错题图片。
    
    Args:
        learner_id: 学习者ID
        image_base64: Base64编码的图片数据
        knowledge_id: 知识点ID（可选）
        user_answer: 用户答案（可选）
        error_type: 错误类型（concept/careless/unknown）
    
    Returns:
        处理结果
    """
    try:
        if not learner_id.strip():
            raise ValueError("learner_id 不能为空")
        if not image_base64.strip():
            raise ValueError("image_base64 不能为空")

        result = request.app.state.orchestrator.upload_wrong_question(
            learner_id=learner_id,
            image_path=None,
            image_base64=image_base64,
            knowledge_id=knowledge_id,
            user_answer=user_answer,
            error_type=error_type
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("upload wrong question base64 failed learner=%s", learner_id)
        raise HTTPException(status_code=500, detail=f"上传错题失败：{str(e)}")


@router.get("/wrong-questions/{learner_id}")
async def get_wrong_questions(
    learner_id: str,
    limit: int = Query(50, description="返回数量限制"),
    request: Request = None
) -> dict[str, Any]:
    """
    获取学习者的错题列表。
    
    Args:
        learner_id: 学习者ID
        limit: 返回数量限制
    
    Returns:
        错题列表
    """
    try:
        if not learner_id.strip():
            raise ValueError("learner_id 不能为空")

        questions = request.app.state.orchestrator.get_wrong_questions(learner_id, limit)
        
        return {
            "learner_id": learner_id,
            "questions": questions,
            "count": len(questions)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("get wrong questions failed learner=%s", learner_id)
        raise HTTPException(status_code=500, detail="获取错题列表失败")


@router.get("/wrong-question/{question_id}")
async def get_wrong_question_detail(
    question_id: int,
    request: Request = None
) -> dict[str, Any]:
    """
    获取错题详情，包括生成的练习题和练习记录。
    
    Args:
        question_id: 错题ID
    
    Returns:
        错题详情
    """
    try:
        detail = request.app.state.orchestrator.get_wrong_question_detail(question_id)
        
        if not detail:
            raise HTTPException(status_code=404, detail="错题不存在")
        
        return detail

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("get wrong question detail failed question_id=%d", question_id)
        raise HTTPException(status_code=500, detail="获取错题详情失败")


@router.post("/wrong-question/{question_id}/practice")
async def practice_wrong_question(
    question_id: int,
    req: WrongQuestionPracticeRequest,
    request: Request = None
) -> dict[str, Any]:
    """
    练习错题，记录答题结果。
    
    Args:
        question_id: 错题ID
        learner_id: 学习者ID
        user_answer: 用户答案
        is_correct: 是否正确
        time_spent: 用时（秒）
    
    Returns:
        练习结果
    """
    try:
        if not req.learner_id.strip():
            raise ValueError("learner_id 不能为空")

        result = request.app.state.orchestrator.practice_wrong_question(
            question_id=question_id,
            learner_id=req.learner_id,
            user_answer=req.user_answer,
            is_correct=req.is_correct,
            time_spent=req.time_spent
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("practice wrong question failed question_id=%d", question_id)
        raise HTTPException(status_code=500, detail="练习错题失败")


@router.delete("/wrong-question/{question_id}")
async def delete_wrong_question(
    question_id: int,
    request: Request = None
) -> dict[str, Any]:
    """
    删除错题。
    
    Args:
        question_id: 错题ID
    
    Returns:
        删除结果
    """
    try:
        success = request.app.state.orchestrator.delete_wrong_question(question_id)
        
        if success:
            return {"success": True, "message": "删除成功"}
        else:
            return {"success": False, "message": "删除失败"}

    except Exception as e:
        logger.exception("delete wrong question failed question_id=%d", question_id)
        raise HTTPException(status_code=500, detail="删除错题失败")


@router.get("/wrong-questions/count/{learner_id}")
async def get_wrong_questions_count(
    learner_id: str,
    request: Request = None
) -> dict[str, Any]:
    """
    获取学习者的错题数量。
    
    Args:
        learner_id: 学习者ID
    
    Returns:
        错题数量
    """
    try:
        if not learner_id.strip():
            raise ValueError("learner_id 不能为空")

        count = request.app.state.orchestrator.get_wrong_questions_count(learner_id)
        
        return {"learner_id": learner_id, "count": count}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("get wrong questions count failed learner=%s", learner_id)
        raise HTTPException(status_code=500, detail="获取错题数量失败")

# ==================== 错题本相关路由结束 ====================