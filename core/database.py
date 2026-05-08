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
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.learner_model import LearnerModel, KnowledgeState, BKTParams

logger = logging.getLogger(__name__)


class Database:
    """SQLite 数据库管理器。"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据库。

        Args:
            db_path: 数据库文件路径
        """
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            db_path = str(project_root / "data" / "edu_agent.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

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
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_learning_history_learner_ts
            ON learning_history (learner_id, timestamp DESC)
        ''')

        # Agent状态持久化（解决服务重启后内存状态丢失）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_name TEXT NOT NULL,
                learner_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (agent_name, learner_id)
            )
        ''')

        # 错题本表（新增）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wrong_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                knowledge_id TEXT NOT NULL,
                original_text TEXT NOT NULL,
                question_text TEXT NOT NULL,
                user_answer TEXT,
                correct_answer TEXT,
                error_type TEXT DEFAULT 'unknown',
                analysis TEXT,
                image_path TEXT,
                created_at TEXT NOT NULL,
                reviewed BOOLEAN DEFAULT FALSE,
                review_count INTEGER DEFAULT 0,
                last_reviewed_at TEXT,
                FOREIGN KEY (learner_id) REFERENCES learner_models(learner_id),
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_states(knowledge_id)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_wrong_questions_learner
            ON wrong_questions (learner_id, created_at DESC)
        ''')

        # 错题练习记录表（新增）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wrong_question_practices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wrong_question_id INTEGER NOT NULL,
                learner_id TEXT NOT NULL,
                practice_time TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                user_answer TEXT,
                time_spent INTEGER,
                FOREIGN KEY (wrong_question_id) REFERENCES wrong_questions(id),
                FOREIGN KEY (learner_id) REFERENCES learner_models(learner_id)
            )
        ''')

        # 错题生成的练习题表（新增）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wrong_question_exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wrong_question_id INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                correct_answer TEXT,
                difficulty INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY (wrong_question_id) REFERENCES wrong_questions(id)
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

    # ------------------ 新增：为了兼容你的调用 ------------------
    def add_learning_history(
        self,
        learner_id: str,
        knowledge_id: str,
        event_type: str,
        is_correct: bool = None,
        mastery: float = None,
        time_spent: int = None
    ):
        """
        兼容式添加学习历史记录（适配 orchestrator 调用）。
        """
        event_data = {
            "is_correct": is_correct,
            "mastery": mastery,
            "time_spent": time_spent
        }
        self.log_learning_event(learner_id, knowledge_id, event_type, event_data)
    # -----------------------------------------------------------

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

    def save_agent_state(self, agent_name: str, learner_id: str, state_data: Dict[str, Any]) -> bool:
        """保存 Agent 的学习者局部状态。"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT OR REPLACE INTO agent_states
                (agent_name, learner_id, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ''',
                (agent_name, learner_id, json.dumps(state_data), datetime.now().isoformat()),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.exception("Failed to save agent state", exc_info=e)
            return False

    def load_agent_state(self, agent_name: str, learner_id: str) -> Optional[Dict[str, Any]]:
        """加载 Agent 的学习者局部状态。"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT state_json FROM agent_states
                WHERE agent_name = ? AND learner_id = ?
                ''',
                (agent_name, learner_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return json.loads(row["state_json"])
        except Exception as e:
            logger.exception("Failed to load agent state", exc_info=e)
            return None

    # ==================== 错题本相关方法（新增）====================

    def add_wrong_question(
        self,
        learner_id: str,
        knowledge_id: str,
        original_text: str,
        question_text: str,
        user_answer: str = None,
        correct_answer: str = None,
        error_type: str = "unknown",
        analysis: str = None,
        image_path: str = None
    ) -> int:
        """
        添加错题记录。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
            original_text: 原始识别文本
            question_text: 提取的题目文本
            user_answer: 用户答案
            correct_answer: 正确答案
            error_type: 错误类型（concept/careless/unknown）
            analysis: 题目解析
            image_path: 图片路径

        Returns:
            int: 错题记录ID
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute('''
                INSERT INTO wrong_questions 
                (learner_id, knowledge_id, original_text, question_text, 
                 user_answer, correct_answer, error_type, analysis, image_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                learner_id,
                knowledge_id,
                original_text,
                question_text,
                user_answer,
                correct_answer,
                error_type,
                analysis,
                image_path,
                now
            ))

            conn.commit()
            logger.info(f"Added wrong question for learner {learner_id}")
            return cursor.lastrowid
        except Exception as e:
            logger.exception("Failed to add wrong question", exc_info=e)
            return -1

    def get_wrong_questions(self, learner_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取学习者的错题列表。

        Args:
            learner_id: 学习者ID
            limit: 返回数量限制

        Returns:
            List[Dict[str, Any]]: 错题列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM wrong_questions 
                WHERE learner_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (learner_id, limit))

            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    "id": row['id'],
                    "knowledge_id": row['knowledge_id'],
                    "original_text": row['original_text'],
                    "question_text": row['question_text'],
                    "user_answer": row['user_answer'],
                    "correct_answer": row['correct_answer'],
                    "error_type": row['error_type'],
                    "analysis": row['analysis'],
                    "image_path": row['image_path'],
                    "created_at": row['created_at'],
                    "reviewed": bool(row['reviewed']),
                    "review_count": row['review_count'],
                    "last_reviewed_at": row['last_reviewed_at']
                })

            return result
        except Exception as e:
            logger.exception("Failed to get wrong questions", exc_info=e)
            return []

    def get_wrong_question_by_id(self, question_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID获取错题记录。

        Args:
            question_id: 错题ID

        Returns:
            Optional[Dict[str, Any]]: 错题记录
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM wrong_questions WHERE id = ?
            ''', (question_id,))

            row = cursor.fetchone()
            if row:
                return {
                    "id": row['id'],
                    "learner_id": row['learner_id'],
                    "knowledge_id": row['knowledge_id'],
                    "original_text": row['original_text'],
                    "question_text": row['question_text'],
                    "user_answer": row['user_answer'],
                    "correct_answer": row['correct_answer'],
                    "error_type": row['error_type'],
                    "analysis": row['analysis'],
                    "image_path": row['image_path'],
                    "created_at": row['created_at'],
                    "reviewed": bool(row['reviewed']),
                    "review_count": row['review_count'],
                    "last_reviewed_at": row['last_reviewed_at']
                }
            return None
        except Exception as e:
            logger.exception("Failed to get wrong question by ID", exc_info=e)
            return None

    def update_wrong_question(
        self,
        question_id: int,
        **kwargs
    ) -> bool:
        """
        更新错题记录。

        Args:
            question_id: 错题ID
            kwargs: 要更新的字段

        Returns:
            bool: 是否更新成功
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            update_fields = []
            update_values = []
            for key, value in kwargs.items():
                if key in ['knowledge_id', 'original_text', 'question_text', 'user_answer',
                           'correct_answer', 'error_type', 'analysis', 'image_path',
                           'reviewed', 'review_count', 'last_reviewed_at']:
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)

            if not update_fields:
                return False

            update_values.append(question_id)
            update_sql = f'''
                UPDATE wrong_questions 
                SET {", ".join(update_fields)} 
                WHERE id = ?
            '''

            cursor.execute(update_sql, update_values)
            conn.commit()
            logger.info(f"Updated wrong question {question_id}")
            return True
        except Exception as e:
            logger.exception("Failed to update wrong question", exc_info=e)
            return False

    def delete_wrong_question(self, question_id: int) -> bool:
        """
        删除错题记录。

        Args:
            question_id: 错题ID

        Returns:
            bool: 是否删除成功
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('DELETE FROM wrong_question_exercises WHERE wrong_question_id = ?', (question_id,))
            cursor.execute('DELETE FROM wrong_question_practices WHERE wrong_question_id = ?', (question_id,))
            cursor.execute('DELETE FROM wrong_questions WHERE id = ?', (question_id,))

            conn.commit()
            logger.info(f"Deleted wrong question {question_id}")
            return True
        except Exception as e:
            logger.exception("Failed to delete wrong question", exc_info=e)
            return False

    def add_wrong_question_practice(
        self,
        wrong_question_id: int,
        learner_id: str,
        is_correct: bool,
        user_answer: str = None,
        time_spent: int = None
    ) -> bool:
        """
        添加错题练习记录。

        Args:
            wrong_question_id: 错题ID
            learner_id: 学习者ID
            is_correct: 是否正确
            user_answer: 用户答案
            time_spent: 用时（秒）

        Returns:
            bool: 是否成功
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO wrong_question_practices 
                (wrong_question_id, learner_id, practice_time, is_correct, user_answer, time_spent)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                wrong_question_id,
                learner_id,
                datetime.now().isoformat(),
                is_correct,
                user_answer,
                time_spent
            ))

            # 更新错题的复习次数和状态
            cursor.execute('''
                UPDATE wrong_questions 
                SET review_count = review_count + 1, 
                    last_reviewed_at = ?,
                    reviewed = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), is_correct, wrong_question_id))

            conn.commit()
            logger.info(f"Added practice record for wrong question {wrong_question_id}")
            return True
        except Exception as e:
            logger.exception("Failed to add wrong question practice", exc_info=e)
            return False

    def get_wrong_question_practices(self, wrong_question_id: int) -> List[Dict[str, Any]]:
        """
        获取错题的练习记录。

        Args:
            wrong_question_id: 错题ID

        Returns:
            List[Dict[str, Any]]: 练习记录列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM wrong_question_practices 
                WHERE wrong_question_id = ? 
                ORDER BY practice_time DESC
            ''', (wrong_question_id,))

            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    "id": row['id'],
                    "practice_time": row['practice_time'],
                    "is_correct": bool(row['is_correct']),
                    "user_answer": row['user_answer'],
                    "time_spent": row['time_spent']
                })

            return result
        except Exception as e:
            logger.exception("Failed to get wrong question practices", exc_info=e)
            return []

    def add_wrong_question_exercise(
        self,
        wrong_question_id: int,
        question_text: str,
        correct_answer: str,
        difficulty: int = 1
    ) -> int:
        """
        添加错题生成的练习题。

        Args:
            wrong_question_id: 错题ID
            question_text: 题目文本
            correct_answer: 正确答案
            difficulty: 难度（1-5）

        Returns:
            int: 练习题ID
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO wrong_question_exercises 
                (wrong_question_id, question_text, correct_answer, difficulty, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                wrong_question_id,
                question_text,
                correct_answer,
                difficulty,
                datetime.now().isoformat()
            ))

            conn.commit()
            logger.info(f"Added exercise for wrong question {wrong_question_id}")
            return cursor.lastrowid
        except Exception as e:
            logger.exception("Failed to add wrong question exercise", exc_info=e)
            return -1

    def get_wrong_question_exercises(self, wrong_question_id: int) -> List[Dict[str, Any]]:
        """
        获取错题生成的练习题。

        Args:
            wrong_question_id: 错题ID

        Returns:
            List[Dict[str, Any]]: 练习题列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM wrong_question_exercises 
                WHERE wrong_question_id = ? 
                ORDER BY difficulty ASC
            ''', (wrong_question_id,))

            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    "id": row['id'],
                    "question_text": row['question_text'],
                    "correct_answer": row['correct_answer'],
                    "difficulty": row['difficulty'],
                    "created_at": row['created_at']
                })

            return result
        except Exception as e:
            logger.exception("Failed to get wrong question exercises", exc_info=e)
            return []

    def get_wrong_questions_count(self, learner_id: str) -> int:
        """
        获取学习者的错题数量。

        Args:
            learner_id: 学习者ID

        Returns:
            int: 错题数量
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT COUNT(*) FROM wrong_questions WHERE learner_id = ?
            ''', (learner_id,))

            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.exception("Failed to get wrong questions count", exc_info=e)
            return 0

    # ==================== 错题本相关方法结束 ====================

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


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