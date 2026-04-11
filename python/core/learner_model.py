"""
学习者模型 -- 贝叶斯知识追踪 (BKT)。
每个学习者有一个持久化的知识状态模型，记录其对每个知识点的掌握程度。
使用贝叶斯知识追踪算法，通过学生的答题表现实时更新mastery概率。
面试要点：
- BKT 四个参数：P(L₀)初始掌握, P(T)转移, P(G)猜测, P(S)失误
- 为什么用Beta分布：能表达不确定性，(alpha, beta)参数对直觉友好
- 与 IRT（项目反应理论）的区别：BKT追踪动态变化，IRT是静态模型
"""
import logging
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from config.settings import settings

logger = logging.getLogger(__name__)


class MasteryLevel(str, Enum):
    """掌握度等级（用于前端展示和课程规划）。"""
    NOT_STARTED = "not_started"  # 未学习
    BEGINNER = "beginner"  # 初学 (0 - 0.3)
    DEVELOPING = "developing"  # 发展中 (0.3 - 0.6)
    PROFICIENT = "proficient"  # 熟练 (0.6 - 0.85)
    MASTERED = "mastered"  # 掌握 (0.85 - 1.0)


class KnowledgeState(BaseModel):
    """单个知识点的掌握状态。"""
    knowledge_id: str
    mastery: float = 0.1  # P(L) 当前掌握概率
    alpha: float = 1.0  # Beta分布参数α（成功次数+1）
    beta: float = 9.0  # Beta分布参数β（失败次数+1）
    attempts: int = 0
    correct_count: int = 0
    last_attempt: datetime | None = None
    streak: int = 0  # 连续正确次数

    @property
    def level(self) -> MasteryLevel:
        """
        根据mastery值获取掌握度等级。

        Returns:
            MasteryLevel: 掌握度等级
        """
        if self.attempts == 0:
            return MasteryLevel.NOT_STARTED
        if self.mastery < 0.3:
            return MasteryLevel.BEGINNER
        if self.mastery < 0.6:
            return MasteryLevel.DEVELOPING
        if self.mastery < 0.85:
            return MasteryLevel.PROFICIENT
        return MasteryLevel.MASTERED

    @property
    def confidence(self) -> float:
        """
        置信度：数据越多，置信度越高。

        Returns:
            float: 置信度值（0-1）
        """
        total = self.alpha + self.beta
        return min(1.0, total / 20.0)


class BKTParams(BaseModel):
    """贝叶斯知识追踪参数（可按知识点/学科调整）。"""
    p_init: float = settings.bkt_p_init  # P(L₀) 初始掌握概率
    p_transit: float = settings.bkt_p_transit  # P(T) 每次练习后学会的概率
    p_guess: float = settings.bkt_p_guess  # P(G) 猜对的概率
    p_slip: float = settings.bkt_p_slip  # P(S) 失误的概率


class LearnerModel:
    """
    学习者知识模型，管理一个学生的所有知识点状态。
    核心方法：
    - update_mastery(): 根据答题结果更新mastery（BKT核心算法）
    - get_weak_points(): 获取薄弱知识点
    - get_ready_topics(): 获取可以学习的新知识点
    """
    def __init__(self, learner_id: str, bkt_params: BKTParams | None = None) -> None:
        """
        初始化学习者模型。

        Args:
            learner_id: 学习者ID
            bkt_params: BKT算法参数，None则使用默认配置
        """
        self.learner_id = learner_id
        self.bkt = bkt_params or BKTParams()
        self.knowledge_states: dict[str, KnowledgeState] = {}
        self.session_start: datetime = datetime.now()
        self.total_interactions: int = 0
        self.metadata: dict[str, Any] = {}

    def get_state(self, knowledge_id: str) -> KnowledgeState:
        """
        获取某知识点的状态，不存在则创建。

        Args:
            knowledge_id: 知识点ID

        Returns:
            KnowledgeState: 知识点状态
        """
        if knowledge_id not in self.knowledge_states:
            self.knowledge_states[knowledge_id] = KnowledgeState(
                knowledge_id=knowledge_id,
                mastery=self.bkt.p_init,
            )
        return self.knowledge_states[knowledge_id]

    def update_mastery(self, knowledge_id: str, is_correct: bool) -> KnowledgeState:
        """
        BKT 核心算法：根据答题结果更新 mastery 概率。
        数学推导：
          P(Lₙ|obs) = P(obs|Lₙ) × P(Lₙ) / P(obs)
          如果答对 (obs = correct):
            P(correct|L) = 1 - P(S)     # 会了且没失误
            P(correct|¬L) = P(G)         # 不会但猜对
            P(Lₙ|correct) = P(L)×(1-P(S)) / [P(L)×(1-P(S)) + (1-P(L))×P(G)]
          如果答错 (obs = wrong):
            P(wrong|L) = P(S)            # 会了但失误
            P(wrong|¬L) = 1 - P(G)      # 不会且没猜对
            P(Lₙ|wrong) = P(L)×P(S) / [P(L)×P(S) + (1-P(L))×(1-P(G))]
          学习转移（每次练习都可能学会）：
            P(Lₙ) = P(Lₙ|obs) + (1 - P(Lₙ|obs)) × P(T)

        Args:
            knowledge_id: 知识点ID
            is_correct: 是否答对

        Returns:
            KnowledgeState: 更新后的知识点状态
        """
        state = self.get_state(knowledge_id)
        p_l = state.mastery

        if is_correct:
            p_obs_given_l = 1 - self.bkt.p_slip
            p_obs_given_not_l = self.bkt.p_guess
            state.correct_count += 1
            state.streak += 1
            state.alpha += 1
        else:
            p_obs_given_l = self.bkt.p_slip
            p_obs_given_not_l = 1 - self.bkt.p_guess
            state.streak = 0
            state.beta += 1

        # 贝叶斯更新
        p_obs = p_l * p_obs_given_l + (1 - p_l) * p_obs_given_not_l
        p_l_given_obs = (p_l * p_obs_given_l) / p_obs if p_obs > 0 else p_l

        # 学习转移
        p_l_new = p_l_given_obs + (1 - p_l_given_obs) * self.bkt.p_transit
        state.mastery = max(0.0, min(1.0, p_l_new))

        state.attempts += 1
        state.last_attempt = datetime.now()
        self.total_interactions += 1

        logger.info(
            "[BKT] Updated mastery: learner=%s, kp=%s, correct=%s, "
            "mastery=%.3f→%.3f, level=%s, attempts=%d, streak=%d",
            self.learner_id, knowledge_id, is_correct,
            p_l, state.mastery, state.level.value, state.attempts, state.streak
        )

        logger.debug(
            "[BKT] Calculation details: p_l=%.3f, p_obs_given_l=%.3f, "
            "p_obs_given_not_l=%.3f, p_obs=%.3f",
            p_l, p_obs_given_l, p_obs_given_not_l, p_obs
        )

        return state

    def get_weak_points(self, threshold: float = 0.4, limit: int = 10) -> list[KnowledgeState]:
        """
        获取薄弱知识点（mastery低于阈值且已尝试过）。

        Args:
            threshold: mastery阈值
            limit: 返回结果数量限制

        Returns:
            list[KnowledgeState]: 薄弱知识点列表，按mastery升序排列
        """
        weak: list[KnowledgeState] = [
            s for s in self.knowledge_states.values()
            if s.mastery < threshold and s.attempts > 0
        ]
        return sorted(weak, key=lambda s: s.mastery)[:limit]

    def get_strong_points(self, threshold: float = 0.85) -> list[KnowledgeState]:
        """
        获取已掌握的知识点。

        Args:
            threshold: mastery阈值

        Returns:
            list[KnowledgeState]: 已掌握的知识点列表
        """
        return [s for s in self.knowledge_states.values() if s.mastery >= threshold]

    def get_overall_progress(self) -> dict[str, Any]:
        """
        获取整体学习进度统计。

        Returns:
            dict[str, Any]: 学习进度统计信息
        """
        states = list(self.knowledge_states.values())
        if not states:
            return {"total": 0, "avg_mastery": 0.0, "level_distribution": {}}

        level_dist: dict[str, int] = {}
        for s in states:
            level = s.level.value
            level_dist[level] = level_dist.get(level, 0) + 1

        return {
            "total_knowledge_points": len(states),
            "avg_mastery": sum(s.mastery for s in states) / len(states),
            "total_attempts": sum(s.attempts for s in states),
            "total_correct": sum(s.correct_count for s in states),
            "accuracy": (
                sum(s.correct_count for s in states) / max(1, sum(s.attempts for s in states))
            ),
            "level_distribution": level_dist,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        序列化为字典（用于持久化）。

        Returns:
            dict[str, Any]: 序列化后的字典
        """
        return {
            "learner_id": self.learner_id,
            "knowledge_states": {
                kid: state.model_dump() for kid, state in self.knowledge_states.items()
            },
            "total_interactions": self.total_interactions,
            "session_start": self.session_start.isoformat(),
            "metadata": self.metadata,
        }