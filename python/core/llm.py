"""
通义千问 LLM 客户端 -- 统一管理 AI 调用。
核心职责：
1. 封装通义千问 API 调用
2. 提供统一的生成接口
3. 支持不同温度参数的调用
"""
import logging
from typing import Optional

from openai import OpenAI
from config.settings import settings

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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            logger.debug("LLM generation successful, prompt length: %d", len(prompt))
            return result
        except Exception as e:
            logger.exception("LLM generation failed", exc_info=e)
            return "抱歉，我现在无法回答你的问题，请稍后再试。"


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