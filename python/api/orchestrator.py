"""
Agent 编排器 -- 完全基于 LangGraph 重构。
核心改进：
1. 用 LangGraph 替代原事件总线，实现全局状态管理和条件路由
2. 保留原 API 接口，完全向后兼容
3. 支持跨会话的短期和长期记忆
4. 简化了 Agent 之间的协作逻辑
"""
from typing import List, Dict, Any

from core.graph import get_learning_graph
from core.learner_model_manager import LearnerModelManager
from core.database import get_database
from core.knowledge_graph import build_sample_math_graph


class AgentOrchestrator:
    """Agent编排器（LangGraph版）。"""

    def __init__(self) -> None:
        """初始化编排器。"""
        self.graph = get_learning_graph()
        self.learner_model_manager = LearnerModelManager()
        self.db = get_database()
        self.knowledge_graph = build_sample_math_graph()

        # 保留原有的Agent实例（用于兼容和辅助功能）
        from agents import (
            AssessmentAgent,
            TutorAgent,
            CurriculumAgent,
            HintAgent,
            EngagementAgent,
        )
        from core.event_bus import EventBus

        self.event_bus = EventBus()
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

    async def submit_answer(
        self, learner_id: str, knowledge_id: str, is_correct: bool, time_spent: float = 0
    ) -> List[Dict[str, Any]]:
        """
        学生提交答案 -> 触发 LangGraph 学习流程。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
            is_correct: 是否正确
            time_spent: 花费时间（秒）

        Returns:
            List[Dict[str, Any]]: 处理结果列表
        """
        # 构建初始状态
        initial_state = {
            "learner_id": learner_id,
            "knowledge_id": knowledge_id,
            "is_correct": is_correct,
            "time_spent_seconds": time_spent,
            "mastery": 0.1,
            "attempts": 0,
            "hint_level": 1,
            "next_action": "assess",
            "context": {}
        }

        # 配置线程ID（用于记忆）
        config = {"configurable": {"thread_id": learner_id}}

        # 运行LangGraph
        result = await self.graph.ainvoke(initial_state, config=config)

        # 转换为原有的事件格式（向后兼容）
        events = []

        # 添加评估完成事件
        events.append({
            "type": "assessment.complete",
            "source": "AssessmentAgent",
            "data": {
                "knowledge_id": knowledge_id,
                "is_correct": is_correct,
                "mastery": result["mastery"],
                "level": self._get_mastery_level(result["mastery"]),
            }
        })

        # 添加教学回复事件
        if result.get("response"):
            events.append({
                "type": "tutor.teaching_response",
                "source": "TutorAgent",
                "data": {
                    "knowledge_id": knowledge_id,
                    "response": result["response"],
                    "teaching_style": "socratic" if not result.get("hint") else "hint",
                }
            })

        # 添加掌握度更新事件
        events.append({
            "type": "assessment.mastery_updated",
            "source": "AssessmentAgent",
            "data": {
                "knowledge_id": knowledge_id,
                "mastery": result["mastery"],
                "level": self._get_mastery_level(result["mastery"]),
            }
        })

        return events

    async def ask_question(
        self, learner_id: str, knowledge_id: str, question: str
    ) -> List[Dict[str, Any]]:
        """
        学生提问 -> 触发 LangGraph 问答流程。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
            question: 问题内容

        Returns:
            List[Dict[str, Any]]: 处理结果列表
        """
        # 构建初始状态
        initial_state = {
            "learner_id": learner_id,
            "knowledge_id": knowledge_id,
            "question": question,
            "is_correct": None,
            "mastery": 0.1,
            "attempts": 0,
            "hint_level": 1,
            "next_action": "assess",
            "context": {}
        }

        # 配置线程ID（用于记忆）
        config = {"configurable": {"thread_id": learner_id}}

        # 运行LangGraph
        result = await self.graph.ainvoke(initial_state, config=config)

        # 转换为原有的事件格式
        events = []

        # 添加评估完成事件
        events.append({
            "type": "assessment.complete",
            "source": "AssessmentAgent",
            "data": {
                "knowledge_id": knowledge_id,
                "question": question,
                "current_mastery": result["mastery"],
                "current_level": self._get_mastery_level(result["mastery"]),
            }
        })

        # 添加教学回复事件
        if result.get("response"):
            events.append({
                "type": "tutor.teaching_response",
                "source": "TutorAgent",
                "data": {
                    "knowledge_id": knowledge_id,
                    "response": result["response"],
                    "teaching_style": "socratic",
                }
            })

        return events

    async def send_message(
        self, learner_id: str, message: str, knowledge_id: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        学生发送消息 -> 触发 LangGraph 对话流程。

        Args:
            learner_id: 学习者ID
            message: 消息内容
            knowledge_id: 知识点ID

        Returns:
            List[Dict[str, Any]]: 处理结果列表
        """
        # 对于通用消息，直接调用TutorAgent
        return await self.ask_question(learner_id, knowledge_id, message)

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

    def _get_mastery_level(self, mastery: float) -> str:
        """
        根据mastery值获取掌握度等级。

        Args:
            mastery: 掌握度值（0-1）

        Returns:
            str: 掌握度等级
        """
        if mastery < 0.3:
            return "beginner"
        elif mastery < 0.6:
            return "developing"
        elif mastery < 0.85:
            return "proficient"
        else:
            return "mastered"

    def get_event_bus_stats(self) -> dict:
        """
        获取事件总线统计信息。

        Returns:
            dict: 统计信息
        """
        return self.event_bus.get_stats()
orchestrator = AgentOrchestrator()
