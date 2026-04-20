"""监控汇总：Agent 事件漏斗等（基于 EventBus 按类型计数）。"""

from __future__ import annotations

from typing import Any, Dict, List

# 竞赛展示用「协作漏斗」：阶段名 -> EventType 字符串（与 EventType.value 一致）
FUNNEL_STAGES: List[tuple[str, List[str]]] = [
    ("1.学生交互", ["student.submission", "student.question", "student.message"]),
    ("2.评估完成", ["assessment.complete"]),
    ("3.掌握度更新", ["assessment.mastery_updated"]),
    ("4.教学与提示", ["tutor.teaching_response", "hint.response", "tutor.hint_needed"]),
    ("5.课程与复习", ["curriculum.path_updated", "curriculum.review_scheduled", "curriculum.next_topic"]),
    ("6.互动调节", ["engagement.alert", "engagement.encouragement", "engagement.pace_adjustment"]),
]


def build_agent_event_funnel(by_type: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    将 EventBus 的 by_type 计数聚合为漏斗各阶段总量。
    说明：这是「按事件类型聚合的协作视图」，非严格因果漏斗；适合答辩展示多 Agent 协作活跃度。
    """
    rows: List[Dict[str, Any]] = []
    for stage_name, keys in FUNNEL_STAGES:
        total = sum(int(by_type.get(k, 0)) for k in keys)
        breakdown = {k: int(by_type.get(k, 0)) for k in keys}
        rows.append({"stage": stage_name, "count": total, "breakdown": breakdown})
    return rows
