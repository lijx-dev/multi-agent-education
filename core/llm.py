"""
通义千问 LLM 客户端 -- 统一管理 AI 调用。
核心职责：
1. 封装通义千问 API 调用
2. 提供统一的生成接口
3. 支持不同温度参数的调用
"""
import logging
import time
from typing import Optional, Literal

from openai import OpenAI
from config.settings import settings
from core.observability import record_llm_call

logger = logging.getLogger(__name__)


class LLMClient:
    """通义千问 LLM 客户端。"""

    def __init__(self):
        """初始化通义千问客户端。"""
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = settings.dashscope_model
        logger.info("LLMClient initialized with model: %s", self.model)

    def generate(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 1000,
            system_prompt: Optional[str] = None
    ) -> str:
        """
        生成文本回复。

        Args:
            prompt: 用户提示词
            temperature: 温度参数（0-1），越高越有创意
            max_tokens: 最大生成token数
            system_prompt: 系统提示词（可选）

        Returns:
            str: 生成的回复文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            logger.debug("LLM generation successful, prompt length: %d", len(prompt))
            record_llm_call((time.perf_counter() - started) * 1000.0, failed=False)
            return result
        except Exception as e:
            logger.exception("LLM generation failed", exc_info=e)
            record_llm_call((time.perf_counter() - started) * 1000.0, failed=True)
            return "抱歉，我现在无法回答你的问题，请稍后再试。"

    def generate_question(
        self,
        knowledge_point: str,
        mastery: float,
        question_type: Literal["选择", "填空", "解答", "自动"] = "自动",
    ) -> str:
        """
        自动出题：基于知识点 + 掌握度生成不同难度题目（选择 / 填空 / 解答）。

        Args:
            knowledge_point: 知识点（建议传知识点名称或ID，LLM会自行理解）
            mastery: 掌握度 0-1
            question_type: 题型（选择/填空/解答/自动）

        Returns:
            str: 可直接在前端展示的题目（Markdown文本）
        """
        # 难度映射：掌握度越低，题目越基础；掌握度越高，题目越综合
        if mastery < 0.3:
            difficulty = "简单"
        elif mastery < 0.6:
            difficulty = "中等"
        else:
            difficulty = "困难"

        if question_type == "自动":
            # 基于掌握度自动选择更合适的题型
            if mastery < 0.35:
                resolved_type: Literal["选择", "填空", "解答"] = "选择"
            elif mastery < 0.7:
                resolved_type = "填空"
            else:
                resolved_type = "解答"
        else:
            resolved_type = question_type

        system_prompt = (
            "你是一位出题专家与资深数学老师，面向中国初中/高中基础阶段。\n"
            "目标：根据知识点与学生掌握度生成一题，难度与题型匹配，题干清晰、可作答、无歧义。\n"
            "输出要求：\n"
            "1) 只输出题目本身（Markdown），不要输出思路、解析、答案。\n"
            "2) 选择题必须给出 A/B/C/D 四个选项，且只有一个正确选项。\n"
            "3) 填空题给出 1-3 个空，使用“____”表示。\n"
            "4) 解答题要有明确的求解目标与必要条件，可分问。\n"
            "5) 题目尽量贴近真实学习场景，避免超纲符号与冷门技巧。"
        )

        user_prompt = (
            f"请围绕知识点：{knowledge_point}\n"
            f"学生掌握度：{mastery:.0%}\n"
            f"目标难度：{difficulty}\n"
            f"题型：{resolved_type}\n"
            "生成 1 道题。"
        )

        # 难度越高，允许更开放的题干；但整体仍需稳定
        temperature = 0.6 if difficulty == "简单" else (0.7 if difficulty == "中等" else 0.8)
        return self.generate(prompt=user_prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=600)


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    获取 LLM 客户端单例。

    Returns:
        LLMClient: LLM 客户端实例
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client