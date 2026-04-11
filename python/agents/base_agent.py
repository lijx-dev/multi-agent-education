"""
Agent 基类 -- 所有Agent的公共接口和行为。
每个Agent都：
1. 有一个名称和角色描述
2. 连接到EventBus，订阅感兴趣的事件
3. 可以发布事件通知其他Agent
4. 有自己的处理逻辑（子类实现）
面试要点：
- 模板方法模式：基类定义骨架，子类实现细节
- 依赖注入：EventBus通过构造函数注入
- 单一职责原则：每个Agent只关注自己的领域
- 基础设施分离：LearnerModelManager独立管理
"""
import logging
from abc import ABC, abstractmethod
from typing import List

from core.event_bus import Event, EventBus, EventType
from core.learner_model_manager import LearnerModelManager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Agent 基类（重构版）。

    改进点：
    - 分离LearnerModelManager，实现基础设施与业务逻辑分离
    - 添加生命周期钩子方法（on_start, on_stop）
    - 统一错误处理机制
    - 更完善的类型注解和文档字符串

    子类需要实现：
    - subscribed_events: 返回自己订阅的事件类型列表
    - handle_event: 处理接收到的事件
    """

    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        learner_model_manager: LearnerModelManager,
    ) -> None:
        """
        初始化Agent基类。

        Args:
            name: Agent名称
            event_bus: 事件总线实例（依赖注入）
            learner_model_manager: 学习者模型管理器（依赖注入）
        """
        self.name = name
        self.event_bus = event_bus
        self.learner_model_manager = learner_model_manager
        self._is_running = False

        self._register_handlers()
        logger.info("[%s] Agent initialized", self.name)

    def _register_handlers(self) -> None:
        """
        注册事件处理器到EventBus。
        模板方法：子类通过subscribed_events声明订阅的事件。
        """
        for event_type in self.subscribed_events:
            self.event_bus.subscribe(event_type, self.handle_event)
            logger.debug("[%s] Subscribed to %s", self.name, event_type.value)

    @property
    @abstractmethod
    def subscribed_events(self) -> List[EventType]:
        """
        子类声明自己订阅哪些事件。

        Returns:
            List[EventType]: 订阅的事件类型列表
        """
        ...

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """
        子类实现事件处理逻辑。

        Args:
            event: 接收到的事件
        """
        ...

    async def emit(self, event_type: EventType, learner_id: str, data: dict) -> None:
        """
        便捷方法：发布事件。

        Args:
            event_type: 事件类型
            learner_id: 学习者ID
            data: 事件数据
        """
        event = Event(
            type=event_type,
            source=self.name,
            learner_id=learner_id,
            data=data,
        )
        await self.event_bus.publish(event)

    async def on_start(self) -> None:
        """
        Agent启动时的生命周期钩子。
        子类可以重写此方法进行初始化操作。
        """
        self._is_running = True
        logger.info("[%s] Agent started", self.name)

    async def on_stop(self) -> None:
        """
        Agent停止时的生命周期钩子。
        子类可以重写此方法进行清理操作。
        """
        self._is_running = False
        logger.info("[%s] Agent stopped", self.name)

    @property
    def is_running(self) -> bool:
        """
        获取Agent运行状态。

        Returns:
            bool: Agent是否正在运行
        """
        return self._is_running