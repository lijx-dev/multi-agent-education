"""
事件驱动总线 -- Mesh架构的核心通信层（重构版）。
所有Agent通过EventBus发布/订阅事件，实现双向异步通信。
改进点：
- 添加事件优先级支持
- 完善全局异常处理
- 添加事件过滤和统计功能
- 更完善的类型注解和文档字符串
面试要点：
- 发布-订阅模式 vs 请求-响应模式
- 异步解耦的优势：松耦合、可扩展、容错
- 事件溯源（Event Sourcing）的基本思想
- 事件优先级的设计思路
"""
import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """事件优先级枚举。"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventType(str, Enum):
    """系统中所有事件类型的枚举定义。"""
    # 学生交互事件
    STUDENT_SUBMISSION = "student.submission"
    STUDENT_QUESTION = "student.question"
    STUDENT_MESSAGE = "student.message"
    # Assessment Agent 事件
    ASSESSMENT_COMPLETE = "assessment.complete"
    MASTERY_UPDATED = "assessment.mastery_updated"
    WEAKNESS_DETECTED = "assessment.weakness_detected"
    # Tutor Agent 事件
    TEACHING_RESPONSE = "tutor.teaching_response"
    HINT_NEEDED = "tutor.hint_needed"
    DIFFICULTY_ADJUSTED = "tutor.difficulty_adjusted"
    # Curriculum Agent 事件
    PATH_UPDATED = "curriculum.path_updated"
    REVIEW_SCHEDULED = "curriculum.review_scheduled"
    NEXT_TOPIC = "curriculum.next_topic"
    # Hint Agent 事件
    HINT_RESPONSE = "hint.response"
    # Engagement Agent 事件
    ENGAGEMENT_ALERT = "engagement.alert"
    ENCOURAGEMENT = "engagement.encouragement"
    PACE_ADJUSTMENT = "engagement.pace_adjustment"


class Event(BaseModel):
    """事件数据模型（增强版）。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    source: str  # 发布事件的Agent名称
    timestamp: datetime = Field(default_factory=datetime.now)
    learner_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None  # 用于追踪事件链
    priority: EventPriority = EventPriority.NORMAL  # 事件优先级


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    异步事件总线（增强版）。

    核心改进：
    1. 支持事件优先级：高优先级事件先处理
    2. 完善的异常处理：单个事件失败不影响其他事件
    3. 事件统计：记录事件发布和处理统计
    4. 事件过滤：支持按多种条件过滤历史事件

    设计原则：
    - 每个Agent注册自己感兴趣的事件类型
    - 发布事件时，总线将事件分发给所有订阅者
    - 支持事件历史记录（Event Sourcing基础）
    """

    def __init__(self) -> None:
        """初始化事件总线。"""
        # 按优先级分组的订阅者
        self._subscribers: Dict[EventType, Dict[EventPriority, List[EventHandler]]] = {}
        self._event_history: List[Event] = []
        self._dead_letters: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

        # 事件统计
        self._stats = {
            "total_published": 0,
            "total_handled": 0,
            "by_type": {},
            "by_source": {},
        }

        logger.info("EventBus initialized with priority support")

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """
        订阅某类事件（支持优先级）。

        Args:
            event_type: 事件类型
            handler: 事件处理器
            priority: 事件优先级
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = {p: [] for p in EventPriority}

        self._subscribers[event_type][priority].append(handler)
        logger.info(
            "Handler subscribed to %s with priority %s",
            event_type.value, priority.name
        )

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        priority: Optional[EventPriority] = None,
    ) -> None:
        """
        取消订阅。

        Args:
            event_type: 事件类型
            handler: 事件处理器
            priority: 优先级，None则从所有优先级中移除
        """
        if event_type not in self._subscribers:
            return

        if priority is not None:
            priorities = [priority]
        else:
            priorities = list(EventPriority)

        for p in priorities:
            if handler in self._subscribers[event_type][p]:
                self._subscribers[event_type][p].remove(handler)
                logger.info(
                    "Handler unsubscribed from %s (priority %s)",
                    event_type.value, p.name
                )

    async def publish(self, event: Event) -> None:
        """
        发布事件到总线（支持优先级）。

        关键设计：
        1. 按优先级从高到低处理事件
        2. 使用asyncio.gather并发通知同一优先级的所有订阅者
        3. 完善的异常处理，单个handler失败不影响其他
        4. 记录事件统计

        Args:
            event: 要发布的事件
        """
        try:
            # 记录事件历史
            async with self._lock:
                self._event_history.append(event)
                self._update_stats(event)

            logger.info(
                "[EventBus] %s -> %s (learner=%s, priority=%s, event_id=%s)",
                event.source, event.type.value, event.learner_id,
                event.priority.name, event.id
            )

            # 获取该事件类型的所有订阅者
            if event.type not in self._subscribers:
                logger.debug("No handlers for event type %s", event.type.value)
                return

            # 按优先级从高到低处理
            for priority in sorted(EventPriority, reverse=True):
                handlers = self._subscribers[event.type][priority]
                if not handlers:
                    continue

                logger.debug(
                    "Processing %d handlers for %s (priority %s)",
                    len(handlers), event.type.value, priority.name
                )

                # 并发处理同一优先级的所有handler
                tasks = [self._safe_handle(handler, event) for handler in handlers]
                await asyncio.gather(*tasks)

        except Exception as e:
            logger.exception(
                "Failed to publish event %s (id=%s) from %s",
                event.type.value, event.id, event.source, exc_info=e
            )

    async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
        """
        安全执行handler，捕获异常避免影响其他订阅者。

        Args:
            handler: 事件处理器
            event: 事件
        """
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                await handler(event)
                self._stats["total_handled"] += 1
                return
            except Exception:
                logger.exception(
                    "Error in handler for event %s from %s (attempt=%d/%d)",
                    event.type.value,
                    event.source,
                    attempt + 1,
                    max_retries + 1,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.05 * (attempt + 1))
        self._dead_letters.append(
            {
                "event_id": event.id,
                "type": event.type.value,
                "source": event.source,
                "learner_id": event.learner_id,
                "timestamp": event.timestamp.isoformat(),
            }
        )

    def _update_stats(self, event: Event) -> None:
        """
        更新事件统计。

        Args:
            event: 事件
        """
        self._stats["total_published"] += 1

        # 按类型统计
        event_type_str = event.type.value
        self._stats["by_type"][event_type_str] = self._stats["by_type"].get(event_type_str, 0) + 1

        # 按来源统计
        source = event.source
        self._stats["by_source"][source] = self._stats["by_source"].get(source, 0) + 1

    def get_history(
        self,
        learner_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        min_priority: Optional[EventPriority] = None,
        limit: int = 50,
    ) -> List[Event]:
        """
        查询事件历史（增强版过滤）。

        Args:
            learner_id: 按学习者ID过滤
            event_type: 按事件类型过滤
            source: 按事件来源过滤
            min_priority: 最低优先级过滤
            limit: 返回结果数量限制

        Returns:
            List[Event]: 符合条件的事件列表
        """
        events = self._event_history

        if learner_id:
            events = [e for e in events if e.learner_id == learner_id]
        if event_type:
            events = [e for e in events if e.type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        if min_priority is not None:
            events = [e for e in events if e.priority >= min_priority]

        return events[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """
        获取事件总线统计信息。

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            **self._stats,
            "total_in_history": len(self._event_history),
            "active_subscriptions": sum(
                len(handlers)
                for type_subs in self._subscribers.values()
                for handlers in type_subs.values()
            ),
            "dead_letter_count": len(self._dead_letters),
        }

    def clear_history(self) -> None:
        """清空事件历史（用于测试）。"""
        self._event_history.clear()
        logger.info("Event history cleared")