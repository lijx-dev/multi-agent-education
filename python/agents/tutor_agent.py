"""
Tutor Agent（教学Agent）-- 苏格拉底式提问教学。
核心职责：
1. 根据学生水平采用苏格拉底式提问，引导而非告知
2. 根据Assessment结果动态调整教学难度
3. 当学生卡住时，请求Hint Agent提供分级提示
面试要点：
- 苏格拉底式教学：不给答案，通过反问让学生自己发现
- Prompt Engineering：针对不同mastery等级设计不同的Prompt模板
- 引导率85%目标：大部分情况只给暗示和引导
"""
import logging
from typing import List

from .base_agent import BaseAgent
from core.event_bus import Event, EventType
from config.settings import settings

logger = logging.getLogger(__name__)


SOCRATIC_PROMPTS = {
    "beginner": (
        "你是一位耐心的数学老师，学生刚开始学习这个知识点。\n"
        "请用最简单的语言和例子帮助学生理解概念。\n"
        "不要直接给答案，而是：\n"
        "1. 先问学生对相关基础概念是否了解\n"
        "2. 用生活化的类比帮助理解\n"
        "3. 给一个最简单的例题让学生尝试"
    ),
    "developing": (
        "你是一位苏格拉底式的数学老师，学生正在学习中。\n"
        "请通过提问引导学生思考：\n"
        "1. 问学生已经知道哪些相关知识\n"
        "2. 引导学生发现问题的关键步骤\n"
        "3. 当学生卡住时，给一个关键提示而非答案"
    ),
    "proficient": (
        "你是一位挑战型的数学老师，学生已经比较熟练。\n"
        "请：\n"
        "1. 提出更深层的思考问题（为什么？还有其他方法吗？）\n"
        "2. 引导学生发现知识点之间的联系\n"
        "3. 给出变式题目拓展思维"
    ),
    "mastered": (
        "你是一位高级数学导师，学生已掌握此知识点。\n"
        "请：\n"
        "1. 引导学生总结归纳方法论\n"
        "2. 布置综合性、跨知识点的挑战题\n"
        "3. 鼓励学生尝试教别人（费曼学习法）"
    ),
}


class TutorAgent(BaseAgent):
    """教学Agent：采用苏格拉底式提问教学法。"""

    def __init__(self, *args, **kwargs):
        """
        初始化教学Agent。
        """
        super().__init__(*args, **kwargs)
        self._student_attempts: dict[str, int] = {}

    @property
    def subscribed_events(self) -> List[EventType]:
        """
        声明订阅的事件类型。

        Returns:
            List[EventType]: 订阅的事件列表
        """
        return [
            EventType.ASSESSMENT_COMPLETE,
            EventType.STUDENT_MESSAGE,
            EventType.HINT_RESPONSE,
            EventType.ENGAGEMENT_ALERT,
        ]

    async def handle_event(self, event: Event) -> None:
        """
        处理接收到的事件。

        Args:
            event: 接收到的事件
        """
        if event.type == EventType.ASSESSMENT_COMPLETE:
            await self._handle_assessment(event)
        elif event.type == EventType.STUDENT_MESSAGE:
            await self._handle_student_message(event)
        elif event.type == EventType.HINT_RESPONSE:
            await self._handle_hint_response(event)
        elif event.type == EventType.ENGAGEMENT_ALERT:
            await self._handle_engagement_alert(event)

    async def _handle_assessment(self, event: Event) -> None:
        """
        根据评估结果调整教学策略。

        Args:
            event: 评估完成事件
        """
        learner_id = event.learner_id
        knowledge_id = event.data.get("knowledge_id", "")
        mastery = event.data.get("mastery", 0.0)
        level = event.data.get("level", "beginner")
        is_correct = event.data.get("is_correct")

        prompt_template = SOCRATIC_PROMPTS.get(level, SOCRATIC_PROMPTS["beginner"])

        if is_correct is False:
            attempt_key = f"{learner_id}:{knowledge_id}"
            attempts = self._student_attempts.get(attempt_key, 0) + 1
            self._student_attempts[attempt_key] = attempts

            if attempts >= 2:
                await self.emit(
                    EventType.HINT_NEEDED,
                    learner_id,
                    {
                        "knowledge_id": knowledge_id,
                        "mastery": mastery,
                        "attempts": attempts,
                        "level": level,
                    },
                )
                return

        response = self._generate_teaching_response(
            knowledge_id, level, mastery, is_correct, event.data.get("question", "")
        )

        await self.emit(
            EventType.TEACHING_RESPONSE,
            learner_id,
            {
                "knowledge_id": knowledge_id,
                "response": response,
                "teaching_style": "socratic",
                "difficulty_level": level,
                "prompt_template_used": level,
            },
        )

    def _generate_teaching_response(
        self,
        knowledge_id: str,
        level: str,
        mastery: float,
        is_correct: bool | None,
        question: str,
    ) -> str:
        """
        生成教学回复。
        实际生产环境中，这里会调用LLM（如GPT-4/MiniMax）。
        当前用模板演示苏格拉底式教学的逻辑框架。

        Args:
            knowledge_id: 知识点ID
            level: 掌握度等级
            mastery: 掌握度值
            is_correct: 是否答对
            question: 学生问题

        Returns:
            str: 教学回复文本
        """
        if is_correct is True:
            return (
                f"很好！你在「{knowledge_id}」上的表现不错。"
                f"当前掌握度：{mastery:.0%}。\n"
                f"让我问你一个更深入的问题：你能用自己的话解释一下这个概念吗？"
                f"或者，你觉得这个知识点和之前学过的哪个知识点有联系？"
            )
        elif is_correct is False:
            return (
                f"没关系，让我们一起来分析「{knowledge_id}」。\n"
                f"先不看答案，我想问你几个问题：\n"
                f"1. 你觉得这道题考查的是什么知识点？\n"
                f"2. 你做题的时候卡在了哪一步？\n"
                f"3. 能不能先试试用最简单的数字代入看看？"
            )
        else:
            return (
                f"好的，关于「{knowledge_id}」，你的问题是：{question}\n"
                f"在我回答之前，让我先问你：\n"
                f"你对这个知识点已经了解了哪些内容？\n"
                f"试着说说你的理解，我们一起看看对不对。"
            )

    async def _handle_student_message(self, event: Event) -> None:
        """
        处理学生消息。

        Args:
            event: 学生消息事件
        """
        learner_id = event.learner_id
        message = event.data.get("message", "")
        knowledge_id = event.data.get("knowledge_id", "general")

        model = self.learner_model_manager.get_or_create_model(learner_id)
        state = model.get_state(knowledge_id)

        response = self._generate_teaching_response(
            knowledge_id, state.level.value, state.mastery, None, message
        )

        await self.emit(
            EventType.TEACHING_RESPONSE,
            learner_id,
            {
                "knowledge_id": knowledge_id,
                "response": response,
                "teaching_style": "socratic",
                "difficulty_level": state.level.value,
            },
        )

    async def _handle_hint_response(self, event: Event) -> None:
        """
        转发Hint Agent的回复给学生。

        Args:
            event: 提示响应事件
        """
        await self.emit(
            EventType.TEACHING_RESPONSE,
            event.learner_id,
            {
                "knowledge_id": event.data.get("knowledge_id", ""),
                "response": event.data.get("hint_text", ""),
                "teaching_style": "hint",
                "hint_level": event.data.get("hint_level", 1),
            },
        )

    async def _handle_engagement_alert(self, event: Event) -> None:
        """
        响应Engagement Agent的警报，调整教学难度。

        Args:
            event: 互动警报事件
        """
        alert_type = event.data.get("alert_type", "")
        if alert_type == "frustration":
            await self.emit(
                EventType.DIFFICULTY_ADJUSTED,
                event.learner_id,
                {
                    "action": "decrease",
                    "reason": "检测到学生挫败感",
                    "message": "我注意到你可能遇到了困难，让我们换一个角度来看这个问题，从更简单的地方开始。",
                },
            )
        elif alert_type == "boredom":
            await self.emit(
                EventType.DIFFICULTY_ADJUSTED,
                event.learner_id,
                {
                    "action": "increase",
                    "reason": "检测到学生可能感到无聊",
                    "message": "看起来这对你来说太简单了！让我给你一个更有挑战性的问题。",
                },
            )