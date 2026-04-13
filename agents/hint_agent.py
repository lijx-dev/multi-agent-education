"""
Hint Agent（提示Agent）-- 分级提示策略。
核心职责：
1. 收到提示请求后，根据学生尝试次数和mastery决定提示级别
2. 三级提示：暗示(Metacognitive) → 引导(Scaffolding) → 直接答案(Targeted)
3. 目标：85%的情况使用暗示或引导，仅15%给直接答案
面试要点：
- 分级提示的教育学依据：Vygotsky的最近发展区(ZPD)理论
- 为什么不直接给答案？研究表明引导式学习的记忆保留率是被动学习的3倍
- 提示级别如何决定？综合考虑尝试次数、mastery、时间消耗
"""
import logging
from typing import List

from .base_agent import BaseAgent
from core.event_bus import Event, EventType
from config.settings import settings

logger = logging.getLogger(__name__)


class HintLevel:
    """提示级别枚举。"""
    METACOGNITIVE = 1  # 暗示：引导学生反思思路
    SCAFFOLDING = 2  # 引导：给出关键步骤提示
    TARGETED = 3  # 直接：给出答案或具体解法


HINT_TEMPLATES = {
    HintLevel.METACOGNITIVE: {
        "description": "元认知暗示 - 引导学生反思自己的思考过程",
        "templates": [
            "想一想，这道题的题目里有哪些关键信息？你有没有全部用上？",
            "你能把这道题和之前学过的哪个类型联系起来吗？",
            "试试从结果反推，如果答案是XX，那过程应该是什么样的？",
            "画个图或者列个表，把已知条件整理一下，也许能看出规律。",
            "这道题有没有更简单的特殊情况？先从特殊情况入手试试。",
        ],
    },
    HintLevel.SCAFFOLDING: {
        "description": "脚手架引导 - 给出关键步骤但不给答案",
        "templates": [
            "这道题的关键是用到{concept}。你记得{concept}的公式/定义吗？",
            "第一步：先{step1}。第二步：然后{step2}。你试试看能不能完成。",
            "提示：答案和{hint_value}有关。试着从这个方向思考。",
            "这道题和{similar_problem}很像。回忆一下那道题是怎么做的？",
        ],
    },
    HintLevel.TARGETED: {
        "description": "直接提示 - 给出具体解法（仅在多次尝试后使用）",
        "templates": [
            "好的，这道题的解法是：{solution_steps}。\n关键点是{key_point}。\n建议你自己重新做一遍，确保理解了。",
        ],
    },
}


class HintAgent(BaseAgent):
    """提示Agent：实现分级提示策略。"""

    def __init__(self, *args, **kwargs):
        """
        初始化提示Agent。
        """
        super().__init__(*args, **kwargs)
        self._hint_history: dict[str, dict[str, int]] = {}

    @property
    def subscribed_events(self) -> List[EventType]:
        """
        声明订阅的事件类型。

        Returns:
            List[EventType]: 订阅的事件列表
        """
        return [EventType.HINT_NEEDED]

    async def handle_event(self, event: Event) -> None:
        """
        处理接收到的事件。

        Args:
            event: 接收到的事件
        """
        if event.type == EventType.HINT_NEEDED:
            await self._provide_hint(event)

    def _determine_hint_level(
        self, learner_id: str, knowledge_id: str, mastery: float, attempts: int
    ) -> int:
        """
        决定提示级别的策略。
        规则：
        - 第1-2次尝试 → Level 1（元认知暗示）
        - 第3-4次尝试 → Level 2（脚手架引导）
        - 第5次及以上 → Level 3（直接提示）
        - mastery极低(< 0.15) 且 attempts >= 3 → 提前升级到 Level 3
        目标：85%使用Level 1-2，15%使用Level 3

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
            mastery: 掌握度值
            attempts: 尝试次数

        Returns:
            int: 提示级别（1-3）
        """
        key = f"{learner_id}:{knowledge_id}"
        hint_count = self._hint_history.get(key, 0)

        if mastery < settings.low_mastery_threshold and attempts >= settings.weakness_min_attempts:
            return HintLevel.TARGETED
        if hint_count <= settings.max_hint_level_1:
            return HintLevel.METACOGNITIVE
        elif hint_count <= settings.max_hint_level_2:
            return HintLevel.SCAFFOLDING
        else:
            return HintLevel.TARGETED

    async def _provide_hint(self, event: Event) -> None:
        """
        生成并发送分级提示。

        Args:
            event: 提示请求事件
        """
        learner_id = event.learner_id
        knowledge_id = event.data.get("knowledge_id", "")
        mastery = event.data.get("mastery", 0.0)
        attempts = event.data.get("attempts", 1)

        level = self._determine_hint_level(learner_id, knowledge_id, mastery, attempts)
        key = f"{learner_id}:{knowledge_id}"
        self._hint_history[key] = self._hint_history.get(key, 0) + 1

        hint_text = self._generate_hint(knowledge_id, level)
        level_names = {
            HintLevel.METACOGNITIVE: "metacognitive",
            HintLevel.SCAFFOLDING: "scaffolding",
            HintLevel.TARGETED: "targeted",
        }

        logger.info(
            "[HintAgent] learner=%s, kp=%s, level=%s (%d/%d)",
            learner_id,
            knowledge_id,
            level_names[level],
            self._hint_history[key],
            attempts,
        )

        await self.emit(
            EventType.HINT_RESPONSE,
            learner_id,
            {
                "knowledge_id": knowledge_id,
                "hint_level": level,
                "hint_level_name": level_names[level],
                "hint_text": hint_text,
                "hint_count": self._hint_history[key],
                "description": HINT_TEMPLATES[level]["description"],
            },
        )

    def _generate_hint(self, knowledge_id: str, level: int) -> str:
        """
        生成提示文本（使用通义千问）。

        Args:
            knowledge_id: 知识点ID
            level: 提示级别

        Returns:
            str: 提示文本
        """
        from core.llm import get_llm_client

        llm = get_llm_client()

        # 基于提示级别的系统提示词
        level_descriptions = {
            HintLevel.METACOGNITIVE: "元认知暗示：引导学生反思自己的思考过程，不要给出具体步骤",
            HintLevel.SCAFFOLDING: "脚手架引导：给出关键步骤但不给答案",
            HintLevel.TARGETED: "直接提示：给出具体解法（仅在多次尝试后使用）"
        }

        system_prompt = (
            f"你是一位数学老师，正在给学生提供提示。\n"
            f"当前提示级别：{level} - {level_descriptions[level]}\n"
            f"知识点：{knowledge_id}\n"
            f"请严格按照提示级别生成回复，不要超出范围。"
        )

        user_prompt = f"请给学生提供一个关于「{knowledge_id}」的提示。"

        # 调用通义千问生成提示
        hint_text = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5
        )

        # 添加前缀
        prefixes = {
            HintLevel.METACOGNITIVE: "💡 元认知提示：",
            HintLevel.SCAFFOLDING: "📝 引导提示：",
            HintLevel.TARGETED: "📖 详细解答："
        }

        return f"{prefixes[level]}\n{hint_text}"