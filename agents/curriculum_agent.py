"""
Curriculum Agent（课程Agent）-- 动态学习路径规划与间隔重复排期。
核心职责：
1. 基于知识图谱拓扑排序规划学习路径
2. 使用SM-2算法安排复习时间
3. 检查前置知识是否达标再推进新内容
面试要点：
- 为什么学习路径要动态生成？每个学生进度不同，固定路径不够个性化
- SM-2 vs Leitner 系统：SM-2更精细，连续调整间隔
- 拓扑排序保证学习顺序：前置知识必须先掌握
"""
import logging
from datetime import datetime
from typing import Any, List

from .base_agent import BaseAgent
from core.event_bus import Event, EventType
from core.database import get_database
from core.knowledge_graph import KnowledgeGraph, build_sample_math_graph
from core.spaced_repetition import SpacedRepetition, ReviewItem
from config.settings import settings

logger = logging.getLogger(__name__)


class CurriculumAgent(BaseAgent):
    """课程Agent：动态规划学习路径 + 间隔重复排期。"""

    def __init__(self, *args, **kwargs):
        """
        初始化课程Agent。
        """
        super().__init__(*args, **kwargs)
        self.knowledge_graph: KnowledgeGraph = build_sample_math_graph()
        self.sr = SpacedRepetition()
        self._review_items: dict[str, dict[str, ReviewItem]] = {}
        self._db = get_database()

    @property
    def subscribed_events(self) -> List[EventType]:
        """
        声明订阅的事件类型。

        Returns:
            List[EventType]: 订阅的事件列表
        """
        return [
            EventType.MASTERY_UPDATED,
            EventType.WEAKNESS_DETECTED,
            EventType.PACE_ADJUSTMENT,
        ]

    async def handle_event(self, event: Event) -> None:
        """
        处理接收到的事件。

        Args:
            event: 接收到的事件
        """
        if event.type == EventType.MASTERY_UPDATED:
            await self._handle_mastery_update(event)
        elif event.type == EventType.WEAKNESS_DETECTED:
            await self._handle_weakness(event)
        elif event.type == EventType.PACE_ADJUSTMENT:
            await self._handle_pace_adjustment(event)

    async def _handle_mastery_update(self, event: Event) -> None:
        """
        mastery更新时，更新复习计划 + 检查是否可以推进新知识点。

        Args:
            event: 掌握度更新事件
        """
        learner_id = event.learner_id
        knowledge_id = event.data.get("knowledge_id", "")
        mastery = float(event.data.get("mastery", 0.0))

        review_item = self._get_review_item(learner_id, knowledge_id)
        was_due = review_item.is_due
        quality = self._quality_from_event(mastery, event.data)
        self.sr.review(review_item, quality)

        if was_due and event.data.get("is_correct") is not None:
            review_item.due_cycle_log.append(
                {
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "is_correct": event.data.get("is_correct"),
                    "quality": quality,
                    "mastery_after": mastery,
                }
            )
            review_item.due_cycle_log = review_item.due_cycle_log[-30:]

        self._persist_review_items(learner_id)

        if mastery >= settings.mastery_threshold:
            await self._check_and_recommend_next(learner_id)

        await self._send_review_schedule(learner_id)

    def _quality_from_event(self, mastery: float, data: dict) -> int:
        """
        综合掌握度、是否答对、耗时与错误类型估计 SM-2 的 quality。

        - 答对：quality 至少为 3；掌握度高、耗时合理时可上调。
        - 答错：整体压低；「粗心」较「概念不清」惩罚更轻。
        """
        is_correct = data.get("is_correct")
        if is_correct is None:
            return self._mastery_to_quality(mastery)

        time_spent = float(data.get("time_spent_seconds") or 0)
        err = str(data.get("error_type") or "unknown").lower()

        base = self._mastery_to_quality(mastery)

        if is_correct:
            q = max(3, min(5, base))
            if mastery >= 0.55 and 0 < time_spent < 30:
                q = min(5, q + 1)
            if mastery >= 0.82:
                q = max(q, 4)
            return max(3, min(5, q))

        q = max(0, min(2, base - 2))
        if err == "careless":
            q = min(3, max(1, q + 2))
        elif err == "concept":
            q = max(0, q - 1)
        return max(0, min(5, q))

    def _mastery_to_quality(self, mastery: float) -> int:
        """
        将mastery概率映射到SM-2的quality评分 (0-5)。

        Args:
            mastery: 掌握度值（0-1）

        Returns:
            int: SM-2质量评分（0-5）
        """
        if mastery >= 0.9:
            return 5
        elif mastery >= 0.75:
            return 4
        elif mastery >= 0.6:
            return 3
        elif mastery >= 0.4:
            return 2
        elif mastery >= 0.2:
            return 1
        return 0

    def _get_review_item(self, learner_id: str, knowledge_id: str) -> ReviewItem:
        """
        获取或创建复习条目。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID

        Returns:
            ReviewItem: 复习条目
        """
        if learner_id not in self._review_items:
            self._review_items[learner_id] = self._load_review_items(learner_id)
        items = self._review_items[learner_id]
        if knowledge_id not in items:
            items[knowledge_id] = ReviewItem(knowledge_id=knowledge_id)
            self._persist_review_items(learner_id)
        return items[knowledge_id]

    def _load_review_items(self, learner_id: str) -> dict[str, ReviewItem]:
        payload = self._db.load_agent_state(self.name, learner_id)
        if not payload:
            return {}
        result: dict[str, ReviewItem] = {}
        raw_items = payload.get("items", {})
        for kid, item_data in raw_items.items():
            try:
                result[kid] = ReviewItem(**item_data)
            except Exception:
                continue
        return result

    def _persist_review_items(self, learner_id: str) -> None:
        items = self._review_items.get(learner_id, {})
        payload = {
            "items": {kid: item.model_dump(mode="json") for kid, item in items.items()}
        }
        self._db.save_agent_state(self.name, learner_id, payload)

    async def _check_and_recommend_next(self, learner_id: str) -> None:
        """
        检查是否有新的可学知识点推荐。

        Args:
            learner_id: 学习者ID
        """
        model = self.learner_model_manager.get_or_create_model(learner_id)
        mastered_ids = {
            s.knowledge_id
            for s in model.knowledge_states.values()
            if s.mastery >= settings.mastery_threshold
        }
        ready = self.knowledge_graph.get_ready_nodes(mastered_ids)

        if ready:
            next_topic = ready[0]
            node = self.knowledge_graph.nodes.get(next_topic)
            await self.emit(
                EventType.NEXT_TOPIC,
                learner_id,
                {
                    "knowledge_id": next_topic,
                    "name": node.name if node else next_topic,
                    "difficulty": node.difficulty if node else 0.5,
                    "reason": "前置知识已掌握，推荐学习新内容",
                    "alternatives": ready[1:4],
                },
            )

    async def _send_review_schedule(self, learner_id: str) -> None:
        """
        发送复习计划。

        Args:
            learner_id: 学习者ID
        """
        items = list(self._review_items.get(learner_id, {}).values())
        due_items = self.sr.get_due_items(items)
        schedule = self.sr.get_study_schedule(items, days_ahead=7)
        upcoming = sorted(items, key=lambda x: x.next_review)[:12]

        def _item_brief(item: ReviewItem) -> dict[str, Any]:
            node = self.knowledge_graph.nodes.get(item.knowledge_id)
            return {
                "knowledge_id": item.knowledge_id,
                "name": node.name if node else item.knowledge_id,
                "next_review": item.next_review.isoformat(timespec="minutes"),
                "interval_days": item.interval_days,
                "easiness_factor": round(item.easiness_factor, 3),
                "repetition": item.repetition,
                "is_due": item.is_due,
            }

        await self.emit(
            EventType.REVIEW_SCHEDULED,
            learner_id,
            {
                "due_now": [
                    {
                        "knowledge_id": item.knowledge_id,
                        "overdue_days": round(item.overdue_days, 2),
                        "name": (
                            self.knowledge_graph.nodes[item.knowledge_id].name
                            if item.knowledge_id in self.knowledge_graph.nodes
                            else item.knowledge_id
                        ),
                    }
                    for item in due_items[:10]
                ],
                "weekly_schedule": schedule,
                "upcoming": [_item_brief(i) for i in upcoming],
                "has_due": len(due_items) > 0,
            },
        )

    def build_review_plan_snapshot(self, learner_id: str) -> dict[str, Any]:
        """供前端/API 拉取的复习计划快照（从 DB 懒加载）。"""
        if learner_id not in self._review_items:
            self._review_items[learner_id] = self._load_review_items(learner_id)
        items = list(self._review_items[learner_id].values())
        due_items = self.sr.get_due_items(items)
        schedule = self.sr.get_study_schedule(items, days_ahead=7)
        upcoming = sorted(items, key=lambda x: x.next_review)[:20]

        def _row(item: ReviewItem) -> dict[str, Any]:
            node = self.knowledge_graph.nodes.get(item.knowledge_id)
            return {
                "knowledge_id": item.knowledge_id,
                "name": node.name if node else item.knowledge_id,
                "easiness_factor": round(item.easiness_factor, 4),
                "interval_days": item.interval_days,
                "repetition": item.repetition,
                "next_review": item.next_review.isoformat(timespec="minutes"),
                "is_due": item.is_due,
                "overdue_days": round(item.overdue_days, 2) if item.is_due else 0.0,
                "due_cycle_log": item.due_cycle_log[-8:],
            }

        return {
            "learner_id": learner_id,
            "due": [_row(i) for i in due_items],
            "weekly_schedule": schedule,
            "upcoming": [_row(i) for i in upcoming],
            "item_count": len(items),
        }

    async def _handle_weakness(self, event: Event) -> None:
        """
        处理薄弱知识点，规划补救路径。

        Args:
            event: 薄弱点检测事件
        """
        learner_id = event.learner_id
        knowledge_id = event.data.get("knowledge_id", "")

        model = self.learner_model_manager.get_or_create_model(learner_id)
        mastered_ids = {
            s.knowledge_id
            for s in model.knowledge_states.values()
            if s.mastery >= settings.mastery_threshold
        }
        remedial_path = self.knowledge_graph.get_learning_path(knowledge_id, mastered_ids)

        await self.emit(
            EventType.PATH_UPDATED,
            learner_id,
            {
                "reason": "weakness_detected",
                "weak_knowledge_id": knowledge_id,
                "remedial_path": remedial_path,
                "message": f"检测到「{knowledge_id}」薄弱，建议先复习前置知识",
            },
        )

    async def _handle_pace_adjustment(self, event: Event) -> None:
        """
        响应Engagement Agent的节奏调整请求。

        Args:
            event: 节奏调整事件
        """
        action = event.data.get("action", "")
        learner_id = event.learner_id
        if action == "slow_down":
            logger.info("[CurriculumAgent] Slowing down pace for learner %s", learner_id)
        elif action == "speed_up":
            logger.info("[CurriculumAgent] Speeding up pace for learner %s", learner_id)