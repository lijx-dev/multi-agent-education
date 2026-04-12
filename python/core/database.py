"""
SQLite 数据库层 -- 长期学习记忆持久化。
核心职责：
1. 学习者模型的持久化和加载
2. 学习历史记录的存储
3. 支持跨会话的记忆检索
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from core.learner_model import LearnerModel, KnowledgeState, BKTParams

logger = logging.getLogger(__name__)


class Database:
    """SQLite 数据库管理器。"""

    def __init__(self, db_path: str = "../data/edu_agent.db"):
        """
        初始化数据库。

        Args:
            db_path: 数据库文件路径
        """
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_tables()
        logger.info("Database initialized at %s", db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """
        获取数据库连接。

        Returns:
            sqlite3.Connection: 数据库连接
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_tables(self):
        """初始化数据库表。"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 学习者模型表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learner_models (
                learner_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                total_interactions INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        ''')

        # 知识点状态表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                knowledge_id TEXT NOT NULL,
                mastery REAL NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                attempts INTEGER NOT NULL,
                correct_count INTEGER NOT NULL,
                streak INTEGER NOT NULL,
                last_attempt TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (learner_id) REFERENCES learner_models(learner_id),
                UNIQUE(learner_id, knowledge_id)
            )
        ''')

        # 学习历史记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                knowledge_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (learner_id) REFERENCES learner_models(learner_id)
            )
        ''')

        conn.commit()
        logger.info("Database tables initialized")

    def save_learner_model(self, model: LearnerModel) -> bool:
        """
        保存学习者模型到数据库。

        Args:
            model: 学习者模型

        Returns:
            bool: 是否成功保存
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            # 保存学习者模型
            cursor.execute('''
                INSERT OR REPLACE INTO learner_models 
                (learner_id, created_at, updated_at, total_interactions, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                model.learner_id,
                model.session_start.isoformat(),
                now,
                model.total_interactions,
                json.dumps(model.metadata)
            ))

            # 保存知识点状态
            for knowledge_id, state in model.knowledge_states.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO knowledge_states 
                    (learner_id, knowledge_id, mastery, alpha, beta, attempts, 
                     correct_count, streak, last_attempt, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model.learner_id,
                    knowledge_id,
                    state.mastery,
                    state.alpha,
                    state.beta,
                    state.attempts,
                    state.correct_count,
                    state.streak,
                    state.last_attempt.isoformat() if state.last_attempt else None,
                    now,
                    now
                ))

            conn.commit()
            logger.debug("Saved learner model: %s", model.learner_id)
            return True
        except Exception as e:
            logger.exception("Failed to save learner model", exc_info=e)
            return False

    def load_learner_model(self, learner_id: str, bkt_params: Optional[BKTParams] = None) -> Optional[LearnerModel]:
        """
        从数据库加载学习者模型。

        Args:
            learner_id: 学习者ID
            bkt_params: BKT参数

        Returns:
            Optional[LearnerModel]: 学习者模型，不存在则返回None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 加载学习者模型
            cursor.execute('SELECT * FROM learner_models WHERE learner_id = ?', (learner_id,))
            model_row = cursor.fetchone()

            if not model_row:
                return None

            model = LearnerModel(learner_id=learner_id, bkt_params=bkt_params)
            model.total_interactions = model_row['total_interactions']
            model.metadata = json.loads(model_row['metadata'])

            # 加载知识点状态
            cursor.execute('SELECT * FROM knowledge_states WHERE learner_id = ?', (learner_id,))
            state_rows = cursor.fetchall()

            for row in state_rows:
                state = KnowledgeState(
                    knowledge_id=row['knowledge_id'],
                    mastery=row['mastery'],
                    alpha=row['alpha'],
                    beta=row['beta'],
                    attempts=row['attempts'],
                    correct_count=row['correct_count'],
                    streak=row['streak'],
                    last_attempt=datetime.fromisoformat(row['last_attempt']) if row['last_attempt'] else None
                )
                model.knowledge_states[row['knowledge_id']] = state

            logger.debug("Loaded learner model: %s", learner_id)
            return model
        except Exception as e:
            logger.exception("Failed to load learner model", exc_info=e)
            return None

    def log_learning_event(self, learner_id: str, knowledge_id: str, event_type: str, event_data: Dict[str, Any]):
        """
        记录学习事件。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
            event_type: 事件类型
            event_data: 事件数据
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO learning_history 
                (learner_id, knowledge_id, event_type, event_data, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                learner_id,
                knowledge_id,
                event_type,
                json.dumps(event_data),
                datetime.now().isoformat()
            ))

            conn.commit()
        except Exception as e:
            logger.exception("Failed to log learning event", exc_info=e)

    def get_learning_history(self, learner_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取学习历史。

        Args:
            learner_id: 学习者ID
            limit: 返回结果数量限制

        Returns:
            List[Dict[str, Any]]: 学习历史列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM learning_history 
                WHERE learner_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (learner_id, limit))

            rows = cursor.fetchall()
            history = []
            for row in rows:
                history.append({
                    "id": row['id'],
                    "knowledge_id": row['knowledge_id'],
                    "event_type": row['event_type'],
                    "event_data": json.loads(row['event_data']),
                    "timestamp": row['timestamp']
                })

            return history
        except Exception as e:
            logger.exception("Failed to get learning history", exc_info=e)
            return []


# 全局单例
_db: Optional[Database] = None


def get_database() -> Database:
    """
    获取数据库单例。

    Returns:
        Database: 数据库实例
    """
    global _db
    if _db is None:
        _db = Database()
    return _db