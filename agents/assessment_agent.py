"""
Assessment Agent（评估Agent）-- 知识点掌握度评估与学习路径诊断。
核心职责：
1. 接收学生答题结果，用BKT算法更新mastery
2. 检测薄弱知识点，通知Curriculum Agent调整路径
3. 定期生成学习报告
面试要点：
- 为什么用BKT而不是简单的正确率？
  正确率是静态的，BKT可以追踪动态学习过程，考虑猜测和失误
- 为什么用Beta分布表示mastery？
  Beta分布是二项分布的共轭先验，更新方便，能表达不确定性
"""
import logging
from typing import List

from .base_agent import BaseAgent
from core.event_bus import Event, EventType, EventPriority
from config.settings import settings

logger = logging.getLogger(__name__)


class AssessmentAgent(BaseAgent):
    """评估Agent：实时追踪学生对每个知识点的掌握程度。"""

    @property
    def subscribed_events(self) -> List[EventType]:
        """
        声明订阅的事件类型。

        Returns:
            List[EventType]: 订阅的事件列表
        """
        return [
            EventType.STUDENT_SUBMISSION,
            EventType.STUDENT_QUESTION,
        ]

    async def handle_event(self, event: Event) -> None:
        """
        处理接收到的事件。

        Args:
            event: 接收到的事件
        """
        if event.type == EventType.STUDENT_SUBMISSION:
            await self._handle_submission(event)
        elif event.type == EventType.STUDENT_QUESTION:
            await self._handle_question(event)

    async def _handle_submission(self, event: Event) -> None:
        """
        处理学生答题提交。
        event.data 结构：
        {
            "knowledge_id": "quadratic_eq",
            "is_correct": true,
            "answer": "x = 2",
            "time_spent_seconds": 45
        }

        Args:
            event: 学生提交事件
        """
        learner_id = event.learner_id
        data = event.data
        knowledge_id = data.get("knowledge_id", "")
        is_correct = data.get("is_correct", False)

        model = self.learner_model_manager.get_or_create_model(learner_id)
        state = model.update_mastery(knowledge_id, is_correct)
        # 👇 新增这一段：根据近期准确率更新学习速度
        overall_progress = model.get_overall_progress()
        recent_accuracy = overall_progress.get("accuracy", 0.5)
        model.update_learning_speed(recent_accuracy)

        # 发布掌握度更新事件（高优先级）
        await self.emit(
            EventType.MASTERY_UPDATED,
            learner_id,
            {
                "knowledge_id": knowledge_id,
                "mastery": state.mastery,
                "level": state.level.value,
                "attempts": state.attempts,
                "streak": state.streak,
                "confidence": state.confidence,
            },
        )

        # 检测薄弱知识点
        if state.mastery < settings.weakness_threshold and state.attempts >= settings.weakness_min_attempts:
            await self.emit(
                EventType.WEAKNESS_DETECTED,
                learner_id,
                {
                    "knowledge_id": knowledge_id,
                    "mastery": state.mastery,
                    "attempts": state.attempts,
                    "suggestion": "需要回到前置知识点复习",
                },
            )

        # 发布评估完成事件
        await self.emit(
            EventType.ASSESSMENT_COMPLETE,
            learner_id,
            {
                "knowledge_id": knowledge_id,
                "is_correct": is_correct,
                "mastery": state.mastery,
                "level": state.level.value,
                "overall_progress": model.get_overall_progress(),
            },
        )

    async def _handle_question(self, event: Event) -> None:
        """
        处理学生提问，分析涉及的知识点。

        Args:
            event: 学生提问事件
        """
        learner_id = event.learner_id
        question = event.data.get("question", "")
        knowledge_id = event.data.get("knowledge_id", "general")

        model = self.learner_model_manager.get_or_create_model(learner_id)
        state = model.get_state(knowledge_id)

        await self.emit(
            EventType.ASSESSMENT_COMPLETE,
            learner_id,
            {
                "knowledge_id": knowledge_id,
                "question": question,
                "current_mastery": state.mastery,
                "current_level": state.level.value,
                "context": "student_question",
            },
        )