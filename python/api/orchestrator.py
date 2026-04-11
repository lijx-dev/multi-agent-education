"""
Agent 编排器 -- 初始化所有Agent并连接到EventBus（重构适配版）。
这是系统的"大脑"，负责：
1. 创建EventBus实例
2. 创建LearnerModelManager实例
3. 初始化5个Agent并注入依赖
4. 提供对外接口供API层调用
"""
from typing import List

from core.event_bus import EventBus, Event, EventType
from core.learner_model_manager import LearnerModelManager
from agents import (
    AssessmentAgent,
    TutorAgent,
    CurriculumAgent,
    HintAgent,
    EngagementAgent,
)


class AgentOrchestrator:
    """Agent编排器（重构适配版）。"""

    def __init__(self) -> None:
        """初始化编排器。"""
        self.event_bus = EventBus()
        self.learner_model_manager = LearnerModelManager()

        # 初始化所有Agent，注入依赖
        self.assessment = AssessmentAgent(
            name="AssessmentAgent",
            event_bus=self.event_bus,
            learner_model_manager=self.learner_model_manager,
        )
        self.tutor = TutorAgent(
            name="TutorAgent",
            event_bus=self.event_bus,
            learner_model_manager=self.learner_model_manager,
        )
        self.curriculum = CurriculumAgent(
            name="CurriculumAgent",
            event_bus=self.event_bus,
            learner_model_manager=self.learner_model_manager,
        )
        self.hint = HintAgent(
            name="HintAgent",
            event_bus=self.event_bus,
            learner_model_manager=self.learner_model_manager,
        )
        self.engagement = EngagementAgent(
            name="EngagementAgent",
            event_bus=self.event_bus,
            learner_model_manager=self.learner_model_manager,
        )

        # 启动所有Agent
        self._start_all_agents()

    def _start_all_agents(self) -> None:
        """启动所有Agent（调用生命周期钩子）。"""
        import asyncio
        agents = [self.assessment, self.tutor, self.curriculum, self.hint, self.engagement]
        for agent in agents:
            asyncio.create_task(agent.on_start())

    async def submit_answer(
        self, learner_id: str, knowledge_id: str, is_correct: bool, time_spent: float = 0
    ) -> List[Event]:
        """
        学生提交答案 -> 触发完整的Agent处理链。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
            is_correct: 是否正确
            time_spent: 花费时间（秒）

        Returns:
            List[Event]: 触发的事件列表
        """
        event = Event(
            type=EventType.STUDENT_SUBMISSION,
            source="api",
            learner_id=learner_id,
            data={
                "knowledge_id": knowledge_id,
                "is_correct": is_correct,
                "time_spent_seconds": time_spent,
            },
        )
        await self.event_bus.publish(event)
        return self.event_bus.get_history(learner_id=learner_id, limit=20)

    async def ask_question(
        self, learner_id: str, knowledge_id: str, question: str
    ) -> List[Event]:
        """
        学生提问 -> 触发Assessment + Tutor处理。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
            question: 问题内容

        Returns:
            List[Event]: 触发的事件列表
        """
        event = Event(
            type=EventType.STUDENT_QUESTION,
            source="api",
            learner_id=learner_id,
            data={"knowledge_id": knowledge_id, "question": question},
        )
        await self.event_bus.publish(event)
        return self.event_bus.get_history(learner_id=learner_id, limit=20)

    async def send_message(
        self, learner_id: str, message: str, knowledge_id: str = "general"
    ) -> List[Event]:
        """
        学生发送消息 -> 触发Tutor对话。

        Args:
            learner_id: 学习者ID
            message: 消息内容
            knowledge_id: 知识点ID

        Returns:
            List[Event]: 触发的事件列表
        """
        event = Event(
            type=EventType.STUDENT_MESSAGE,
            source="api",
            learner_id=learner_id,
            data={"message": message, "knowledge_id": knowledge_id},
        )
        await self.event_bus.publish(event)
        return self.event_bus.get_history(learner_id=learner_id, limit=20)

    def get_learner_progress(self, learner_id: str) -> dict:
        """
        获取学习者进度。

        Args:
            learner_id: 学习者ID

        Returns:
            dict: 学习进度信息
        """
        model = self.learner_model_manager.get_model(learner_id)
        if not model:
            return {"learner_id": learner_id, "status": "no_data"}

        return {
            "learner_id": learner_id,
            "progress": model.get_overall_progress(),
            "weak_points": [
                {"id": s.knowledge_id, "mastery": s.mastery}
                for s in model.get_weak_points()
            ],
            "strong_points": [
                {"id": s.knowledge_id, "mastery": s.mastery}
                for s in model.get_strong_points()
            ],
        }

    def get_event_bus_stats(self) -> dict:
        """
        获取事件总线统计信息。

        Returns:
            dict: 统计信息
        """
        return self.event_bus.get_stats()