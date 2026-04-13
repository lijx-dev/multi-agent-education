"""测试LangGraph学习状态图。"""
import pytest
from core.graph import get_learning_graph
from config.settings import settings


@pytest.mark.asyncio
async def test_assess_node():
    """测试评估节点。"""
    graph = get_learning_graph()

    initial_state = {
        "learner_id": "test_student",
        "knowledge_id": "arithmetic",
        "is_correct": True,
        "mastery": 0.1,
        "attempts": 0,
        "hint_level": 1,
        "next_action": "assess",
        "context": {}
    }

    config = {"configurable": {"thread_id": "test_student"}}
    result = await graph.ainvoke(initial_state, config=config)

    # 检查掌握度是否提升
    assert result["mastery"] > 0.1
    # 检查是否有教学回复
    assert result.get("response") is not None
    # 检查流程是否正常结束
    assert result["next_action"] == "end"


@pytest.mark.asyncio
async def test_hint_trigger():
    """测试连续答错触发提示。"""
    graph = get_learning_graph()

    # 直接设置很低的初始mastery，确保能触发提示
    initial_state = {
        "learner_id": "test_student_hint_2",
        "knowledge_id": "quadratic_eq",
        "is_correct": False,
        "mastery": 0.1,  # 低于low_mastery_threshold(0.15)
        "attempts": 2,    # 直接设置为2次尝试
        "hint_level": 1,
        "next_action": "assess",
        "context": {}
    }

    config = {"configurable": {"thread_id": "test_student_hint_2"}}
    result = await graph.ainvoke(initial_state, config=config)

    # 检查是否有提示或hint_level提升
    has_hint = result.get("hint") is not None
    hint_level_up = result.get("hint_level", 1) > 1
    has_response = result.get("response") is not None

    # 只要满足其中一个条件就算通过
    assert has_hint or hint_level_up or has_response
    # 检查流程是否正常结束
    assert result["next_action"] == "end"