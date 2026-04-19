"""
学习者模型管理器 -- 分离自BaseAgent的基础设施。
负责学习者模型的创建、获取和持久化，实现业务逻辑与基础设施的分离。
新增功能：
- 支持从SQLite加载和保存学习者模型
- 支持跨会话的长期记忆
"""
import logging
from typing import Dict, Optional

from core.learner_model import LearnerModel, BKTParams
from core.database import get_database
from config.settings import settings

logger = logging.getLogger(__name__)


class LearnerModelManager:
    """
    学习者模型管理器（增强版）。

    职责：
    - 创建和管理学习者模型实例
    - 提供统一的模型访问接口
    - 支持模型的持久化和加载（SQLite）
    """

    def __init__(self, bkt_params: Optional[BKTParams] = None) -> None:
        """
        初始化学习者模型管理器。

        Args:
            bkt_params: BKT算法参数，None则使用默认配置
        """
        self._learner_models: Dict[str, LearnerModel] = {}
        self._default_bkt_params = bkt_params or BKTParams(
            p_init=settings.bkt_p_init,
            p_transit=settings.bkt_p_transit,
            p_guess=settings.bkt_p_guess,
            p_slip=settings.bkt_p_slip,
        )
        self._db = get_database()
        logger.info("LearnerModelManager initialized with persistence")

    def get_or_create_model(self, learner_id: str) -> LearnerModel:
        """
        获取学习者模型，不存在则创建或从数据库加载。

        Args:
            learner_id: 学习者ID

        Returns:
            LearnerModel: 学习者模型实例
        """
        if learner_id not in self._learner_models:
            # 先尝试从数据库加载
            model = self._db.load_learner_model(learner_id, self._default_bkt_params)
            if model:
                self._learner_models[learner_id] = model
                logger.debug("Loaded LearnerModel from DB: %s", learner_id)
            else:
                # 创建新模型
                self._learner_models[learner_id] = LearnerModel(
                    learner_id=learner_id,
                    bkt_params=self._default_bkt_params,
                )
                logger.debug("Created new LearnerModel: %s", learner_id)

        return self._learner_models[learner_id]

    def save_model(self, learner_id: str) -> bool:
        """
        保存学习者模型到数据库。

        Args:
            learner_id: 学习者ID

        Returns:
            bool: 是否成功保存
        """
        if learner_id in self._learner_models:
            return self._db.save_learner_model(self._learner_models[learner_id])
        return False

    def get_model(self, learner_id: str) -> Optional[LearnerModel]:
        """
        获取已存在的学习者模型（不自动创建）。

        Args:
            learner_id: 学习者ID

        Returns:
            Optional[LearnerModel]: 学习者模型实例，不存在则返回None
        """
        model = self._learner_models.get(learner_id)
        if model:
            return model
        # 未命中内存时尝试从数据库懒加载，避免跨模块实例不一致导致“查不到进度”
        loaded = self._db.load_learner_model(learner_id, self._default_bkt_params)
        if loaded:
            self._learner_models[learner_id] = loaded
        return loaded

    def remove_model(self, learner_id: str) -> bool:
        """
        移除学习者模型。

        Args:
            learner_id: 学习者ID

        Returns:
            bool: 是否成功移除
        """
        if learner_id in self._learner_models:
            del self._learner_models[learner_id]
            logger.info("Removed LearnerModel: %s", learner_id)
            return True
        return False

    def get_all_learner_ids(self) -> list[str]:
        """
        获取所有学习者ID。

        Returns:
            list[str]: 学习者ID列表
        """
        return list(self._learner_models.keys())


_learner_model_manager_singleton: Optional[LearnerModelManager] = None


def get_learner_model_manager() -> LearnerModelManager:
    """获取 LearnerModelManager 全局单例，保证全项目读写同一份学生状态。"""
    global _learner_model_manager_singleton
    if _learner_model_manager_singleton is None:
        _learner_model_manager_singleton = LearnerModelManager()
    return _learner_model_manager_singleton