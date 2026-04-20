"""
SM-2 间隔重复算法（SuperMemo 2）实现。

核心行为（与常见 Anki/SM-2 描述一致）：
1. 根据回答质量 q∈[0,5] 更新难度因子 EF（easiness_factor），并限制在 [MIN_EF, MAX_EF]。
2. q < 3：重置 repetition，interval 置为 1 天（需尽快复习）。
3. q ≥ 3：若 repetition 为 0 → interval=1；为 1 → interval=6；否则
   interval = round(上次间隔 × EF × 遗忘修正因子)。
4. repetition 仅在 q ≥ 3 时 +1。

扩展：
- quality_history：保留近期质量，用于遗忘修正因子（可继续接 Ebbinghaus 等模型）。
- get_due_items / get_study_schedule：到期项与未来日程。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _default_next_review() -> datetime:
    """新建条目尚未进入 SM-2 前，默认为次日复习，避免「未练习即全部到期」。"""
    return datetime.now() + timedelta(days=1)


class ReviewItem(BaseModel):
    """一个知识点的 SM-2 复习条目（可持久化）。"""

    knowledge_id: str
    easiness_factor: float = 2.5
    interval_days: float = 1.0
    repetition: int = 0
    next_review: datetime = Field(default_factory=_default_next_review)
    last_review: datetime | None = None
    total_reviews: int = 0
    # 近期回答质量，用于遗忘修正因子（保留最近若干次即可）
    quality_history: List[int] = Field(default_factory=list)
    # 在「已到期」状态下完成的一次练习后的简要记录（验证间隔重复效果）
    due_cycle_log: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def is_due(self) -> bool:
        # 尚未产生任何 SM-2 记录时，不视为系统意义上的「到期复习」
        if self.total_reviews == 0:
            return False
        return datetime.now() >= self.next_review

    @property
    def overdue_days(self) -> float:
        delta = datetime.now() - self.next_review
        return delta.total_seconds() / 86400


class SpacedRepetition:
    """SM-2 调度器。"""

    MIN_EF = 1.3
    MAX_EF = 2.5
    QUALITY_HISTORY_MAX = 24

    def review(self, item: ReviewItem, quality: int) -> ReviewItem:
        """
        根据 SM-2 规则更新 EF、间隔与下次复习时间。

        Args:
            item: 复习条目（会被原地更新）
            quality: 回答质量 0-5（SuperMemo 语义）

        Returns:
            同一 ReviewItem 实例（便于调用方继续持久化）
        """
        q = max(0, min(5, int(quality)))
        item.total_reviews += 1
        item.last_review = datetime.now()

        # --- 1) 更新 EF（难度因子）---
        # EF' = EF + (0.1 - (5-q)(0.08 + (5-q)*0.02))
        delta_ef = 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
        new_ef = item.easiness_factor + delta_ef
        item.easiness_factor = max(self.MIN_EF, min(self.MAX_EF, new_ef))

        old_rep = item.repetition
        prev_interval = float(item.interval_days)

        # --- 2) 根据 q 更新 repetition 与 interval ---
        if q < 3:
            item.repetition = 0
            item.interval_days = 1.0
        else:
            if old_rep == 0:
                item.interval_days = 1.0
            elif old_rep == 1:
                item.interval_days = 6.0
            else:
                forgetting = self._calculate_forgetting_factor(item)
                item.interval_days = max(
                    1.0,
                    round(prev_interval * item.easiness_factor * forgetting, 4),
                )
            item.repetition = old_rep + 1

        # --- 3) 记录质量轨迹 ---
        item.quality_history.append(q)
        if len(item.quality_history) > self.QUALITY_HISTORY_MAX:
            item.quality_history = item.quality_history[-self.QUALITY_HISTORY_MAX :]

        # --- 4) 下次复习时间 ---
        item.next_review = datetime.now() + timedelta(days=float(item.interval_days))

        logger.info(
            "[SM-2] kp=%s q=%d EF=%.3f interval=%.2f rep=%d next=%s",
            item.knowledge_id,
            q,
            item.easiness_factor,
            item.interval_days,
            item.repetition,
            item.next_review.isoformat(timespec="minutes"),
        )
        return item

    def _calculate_forgetting_factor(self, item: ReviewItem) -> float:
        """
        遗忘曲线修正因子（预留扩展口）：根据近期回答质量微调间隔倍数。

        当前实现：用最近若干次 quality 的均值映射到约 [0.82, 1.18]，
        质量好 → 略拉长间隔；质量差 → 略缩短间隔。
        后续可替换为基于逾期天数、题目难度、个人遗忘曲线的估计。
        """
        hist = item.quality_history[-8:]
        if len(hist) < 2:
            return 1.0
        avg = sum(hist) / len(hist)
        # avg∈[0,5]，以 3 为中心
        t = (avg - 3.0) / 2.0
        factor = 1.0 + 0.18 * max(-1.0, min(1.0, t))
        return max(0.82, min(1.18, factor))

    def get_due_items(self, items: List[ReviewItem]) -> List[ReviewItem]:
        """获取已到期复习项，按逾期程度从高到低排序。"""
        due = [it for it in items if it.is_due]
        return sorted(due, key=lambda x: x.overdue_days, reverse=True)

    def get_study_schedule(
        self, items: List[ReviewItem], days_ahead: int = 7
    ) -> Dict[str, List[str]]:
        """
        生成未来 days_ahead 天内、按「本地日历日」分桶的复习计划。

        使用 [day_start, day_end) 半开区间，避免边界漏项。
        """
        schedule: Dict[str, List[str]] = {}
        now = datetime.now()
        for day_offset in range(days_ahead):
            day_start = (now + timedelta(days=day_offset)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            day_end = day_start + timedelta(days=1)
            date_key = day_start.strftime("%Y-%m-%d")
            day_ids: List[str] = []
            for item in items:
                if day_start <= item.next_review < day_end:
                    day_ids.append(item.knowledge_id)
            if day_ids:
                schedule[date_key] = sorted(set(day_ids))
        return schedule
