"""
LangGraph 学习状态图 -- 替代原事件总线的核心编排。
核心改进：
1. 全局状态管理：所有Agent共享同一个状态图
2. 条件路由：根据学生状态自动选择下一步动作
3. 记忆系统：支持跨会话的短期和长期记忆
4. 工具调用：预留工具调用接口
"""
import logging
from typing import TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.learner_model_manager import LearnerModelManager, get_learner_model_manager
from core.llm import get_llm_client
from core.database import get_database
from config.settings import settings

logger = logging.getLogger(__name__)


class LearningState(TypedDict):
    """全局学习状态图定义。"""
    learner_id: str
    knowledge_id: str
    question: Optional[str]
    answer: Optional[str]
    is_correct: Optional[bool]
    mastery: float
    attempts: int
    hint_level: int
    next_action: str
    response: Optional[str]
    hint: Optional[str]
    context: Dict[str, Any]


# 全局依赖
_learner_manager: Optional[LearnerModelManager] = None
_llm = None
_db = None


def _init_dependencies():
    """初始化全局依赖。"""
    global _learner_manager, _llm, _db
    if _learner_manager is None:
        _learner_manager = get_learner_model_manager()
    if _llm is None:
        _llm = get_llm_client()
    if _db is None:
        _db = get_database()


def assess_node(state: LearningState) -> LearningState:
    """
    评估节点：更新掌握度，检测薄弱点。

    Args:
        state: 当前学习状态

    Returns:
        LearningState: 更新后的状态
    """
    _init_dependencies()

    learner_id = state["learner_id"]
    knowledge_id = state["knowledge_id"]
    is_correct = state.get("is_correct")
    current_attempts = state.get("attempts", 0)

    model = _learner_manager.get_or_create_model(learner_id)

    if is_correct is not None:
        # 更新掌握度
        state_obj = model.update_mastery(knowledge_id, is_correct)
        mastery = state_obj.mastery
        attempts = state_obj.attempts

        # 记录学习事件
        _db.log_learning_event(
            learner_id, knowledge_id, "submission",
            {"is_correct": is_correct, "mastery": mastery}
        )

        # 保存模型
        _learner_manager.save_model(learner_id)
    else:
        # 只是提问，不更新掌握度
        state_obj = model.get_state(knowledge_id)
        mastery = state_obj.mastery
        attempts = state_obj.attempts

    # 决定下一步动作
    if attempts >= 2 and mastery < settings.low_mastery_threshold:
        next_action = "hint"
    elif is_correct is not None:
        next_action = "teach"
    else:
        next_action = "teach"

    return {
        **state,
        "mastery": mastery,
        "attempts": attempts,
        "next_action": next_action
    }

def teach_node(state: LearningState) -> LearningState:
    """
    教学节点：生成苏格拉底式教学回复。

    Args:
        state: 当前学习状态

    Returns:
        LearningState: 更新后的状态
    """
    _init_dependencies()

    knowledge_id = state["knowledge_id"]
    mastery = state["mastery"]
    is_correct = state.get("is_correct")
    question = state.get("question", "")
    context = state.get("context", {})
    chat_history = context.get("chat_history", [])
    recent_history = chat_history[-20:] if isinstance(chat_history, list) else []
    history_text = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_history
    )

    # 基于掌握度选择系统提示词
    if mastery < 0.3:
        level = "beginner"
    elif mastery < 0.6:
        level = "developing"
    elif mastery < 0.85:
        level = "proficient"
    else:
        level = "mastered"

    system_prompts = {
        "beginner": (
            "你是一位耐心的数学老师，学生刚开始学习这个知识点。\n"
            "请用最简单的语言和例子帮助学生理解概念。\n"
            "不要直接给答案，而是通过提问引导学生思考。"
        ),
        "developing": (
            "你是一位苏格拉底式的数学老师，学生正在学习中。\n"
            "请通过提问引导学生思考，不要直接给答案。"
        ),
        "proficient": (
            "你是一位挑战型的数学老师，学生已经比较熟练。\n"
            "请提出更深层的思考问题，引导学生发现知识点之间的联系。"
        ),
        "mastered": (
            "你是一位高级数学导师，学生已掌握此知识点。\n"
            "请引导学生总结归纳方法论，布置综合性挑战题。"
        ),
    }

    system_prompt = system_prompts[level]

    # 构建用户提示词
    if is_correct is True:
        user_prompt = (
            f"学生在「{knowledge_id}」知识点上答对了题目。\n"
            f"当前掌握度：{mastery:.0%}。\n"
            f"请给出鼓励，并提出一个更深入的思考问题。"
        )
    elif is_correct is False:
        user_prompt = (
            f"学生在「{knowledge_id}」知识点上答错了题目。\n"
            f"当前掌握度：{mastery:.0%}。\n"
            f"请不要直接告诉答案，而是通过提问引导学生分析错误原因。"
        )
    else:
        user_prompt = (
            f"学生关于「{knowledge_id}」的问题是：{question}\n"
            f"最近对话（最多10轮）：\n{history_text if history_text else '（无历史）'}\n"
            f"请通过提问引导学生自己思考，不要直接给答案。"
        )

    # 调用LLM生成回复
    response = _llm.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.7
    )

    return {
        **state,
        "response": response,
        "next_action": "end"
    }


def hint_node(state: LearningState) -> LearningState:
    """
    提示节点：生成分级提示。

    Args:
        state: 当前学习状态

    Returns:
        LearningState: 更新后的状态
    """
    _init_dependencies()

    knowledge_id = state["knowledge_id"]
    mastery = state["mastery"]
    attempts = state["attempts"]
    hint_level = state.get("hint_level", 1)

    # 决定提示级别
    if mastery < settings.low_mastery_threshold and attempts >= 3:
        current_level = 3
    elif hint_level <= 1:
        current_level = 1
    elif hint_level <= 3:
        current_level = 2
    else:
        current_level = 3

    # 基于提示级别的系统提示词
    level_descriptions = {
        1: "元认知暗示：引导学生反思自己的思考过程，不要给出具体步骤",
        2: "脚手架引导：给出关键步骤但不给答案",
        3: "直接提示：给出具体解法（仅在多次尝试后使用）"
    }

    system_prompt = (
        f"你是一位数学老师，正在给学生提供提示。\n"
        f"当前提示级别：{current_level} - {level_descriptions[current_level]}\n"
        f"知识点：{knowledge_id}\n"
        f"学生掌握度：{mastery:.0%}\n"
        f"尝试次数：{attempts}\n"
    )

    user_prompt = f"请给学生提供一个关于「{knowledge_id}」的提示。"

    # 调用LLM生成提示
    hint_text = _llm.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.5
    )

    return {
        **state,
        "hint": hint_text,
        "hint_level": current_level + 1,
        "response": hint_text,
        "next_action": "end"
    }


def router(state: LearningState) -> str:
    """
    条件路由：根据状态决定下一步。

    Args:
        state: 当前学习状态

    Returns:
        str: 下一个节点名称
    """
    next_action = state["next_action"]
    if next_action == "hint":
        return "hint"
    elif next_action == "teach":
        return "teach"
    return "end"


# 构建LangGraph
def build_learning_graph():
    """
    构建学习状态图。

    Returns:
        CompiledGraph: 编译后的LangGraph
    """
    builder = StateGraph(LearningState)

    # 添加节点
    builder.add_node("assess", assess_node)
    builder.add_node("teach", teach_node)
    builder.add_node("hint", hint_node)

    # 设置入口点
    builder.set_entry_point("assess")

    # 添加条件边
    builder.add_conditional_edges(
        "assess",
        router,
        {
            "teach": "teach",
            "hint": "hint",
            "end": END
        }
    )

    # 添加结束边
    builder.add_edge("teach", END)
    builder.add_edge("hint", END)

    # 添加记忆
    memory = MemorySaver()

    # 编译图
    return builder.compile(checkpointer=memory)


# 全局单例
_learning_graph = None


def get_learning_graph():
    """
    获取学习状态图单例。

    Returns:
        CompiledGraph: 编译后的LangGraph
    """
    global _learning_graph
    if _learning_graph is None:
        _learning_graph = build_learning_graph()
    return _learning_graph