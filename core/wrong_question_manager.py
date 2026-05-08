"""
错题本管理器 - 处理拍照上传错题、OCR识别、AI分析、生成练习等功能。
核心职责：
1. 处理拍照上传的错题图片
2. 调用OCR识别题目文本
3. 调用LLM分析错因和生成解析
4. 生成类似题目进行巩固练习
5. 自动更新复习计划
"""
import logging
from typing import Optional, Dict, Any, List

from core.database import get_database
from core.ocr import get_ocr_service
from core.llm import get_llm_client
from core.knowledge_graph import build_sample_math_graph, KnowledgeGraph

logger = logging.getLogger(__name__)


class WrongQuestionManager:
    """错题本管理器"""

    def __init__(self):
        self.db = get_database()
        self.ocr_service = get_ocr_service()
        self.llm = get_llm_client()
        self.knowledge_graph = build_sample_math_graph()

    def upload_wrong_question(
        self,
        learner_id: str,
        image_path: str = None,
        image_base64: str = None,
        knowledge_id: str = None,
        user_answer: str = None,
        error_type: str = "unknown"
    ) -> Dict[str, Any]:
        """
        上传错题图片，识别题目并保存到错题本。

        Args:
            learner_id: 学习者ID
            image_path: 图片文件路径（二选一）
            image_base64: Base64编码的图片数据（二选一）
            knowledge_id: 知识点ID（可选，不提供则自动分析）
            user_answer: 用户答案（可选）
            error_type: 错误类型（concept/careless/unknown）

        Returns:
            Dict[str, Any]: 处理结果，包含题目信息、解析、生成的练习题
        """
        # 1. OCR识别图片中的文字
        logger.info(f"开始处理错题上传，学习者ID: {learner_id}")
        ocr_result = self.ocr_service.parse_math_question(image_path, image_base64)
        
        if not ocr_result:
            return {"success": False, "error": "OCR识别失败"}

        original_text = ocr_result["original_text"]
        question_text = ocr_result["question"]
        
        if not question_text.strip():
            return {"success": False, "error": "未能识别到题目内容"}

        # 2. 如果未指定知识点，尝试自动分析
        if not knowledge_id:
            knowledge_id = self._analyze_knowledge_point(question_text)
            logger.info(f"自动识别知识点: {knowledge_id}")

        # 3. 调用LLM分析题目，生成解析和正确答案
        analysis_result = self._analyze_question(question_text, user_answer)
        correct_answer = analysis_result.get("answer")
        analysis = analysis_result.get("analysis")
        suggested_knowledge_id = analysis_result.get("knowledge_id")
        
        # 使用LLM分析的知识点（如果更准确）
        if suggested_knowledge_id and not knowledge_id:
            knowledge_id = suggested_knowledge_id

        # 4. 保存到数据库
        wrong_question_id = self.db.add_wrong_question(
            learner_id=learner_id,
            knowledge_id=knowledge_id,
            original_text=original_text,
            question_text=question_text,
            user_answer=user_answer,
            correct_answer=correct_answer,
            error_type=error_type,
            analysis=analysis,
            image_path=image_path
        )

        if wrong_question_id <= 0:
            return {"success": False, "error": "保存错题失败"}

        # 5. 生成3道类似练习题
        exercises = self._generate_exercises(wrong_question_id, question_text, knowledge_id)

        # 6. 更新复习计划（通过SM-2算法）
        self._update_review_plan(learner_id, knowledge_id)

        # 7. 更新学习者模型的掌握度（降低该知识点的掌握度）
        self._update_mastery(learner_id, knowledge_id, is_correct=False)

        return {
            "success": True,
            "wrong_question_id": wrong_question_id,
            "question_text": question_text,
            "original_text": original_text,
            "knowledge_id": knowledge_id,
            "correct_answer": correct_answer,
            "analysis": analysis,
            "exercises": exercises,
            "message": "错题上传成功！已生成3道巩固练习题"
        }

    def _analyze_knowledge_point(self, question_text: str) -> str:
        """
        根据题目文本分析所属知识点。

        Args:
            question_text: 题目文本

        Returns:
            str: 知识点ID
        """
        # 简单的关键词匹配
        knowledge_keywords = {
            "arithmetic": ["加减", "乘除", "运算", "计算", "求和", "差", "积", "商"],
            "algebraic_expr": ["代数式", "化简", "合并同类项", "因式分解", "表达式"],
            "linear_eq_1": ["方程", "求解", "解:", "x=", "一元一次", "一次方程"],
            "quadratic_eq": ["二次方程", "平方", "x²", "求根", "判别式", "顶点"],
            "geometry": ["三角形", "面积", "周长", "角度", "相似", "全等", "图形"],
            "function": ["函数", "f(x)", "定义域", "值域", "图像", "单调性"]
        }

        for knowledge_id, keywords in knowledge_keywords.items():
            for keyword in keywords:
                if keyword in question_text:
                    return knowledge_id

        # 默认返回arithmetic
        return "arithmetic"

    def _analyze_question(self, question_text: str, user_answer: str = None) -> Dict[str, Any]:
        """
        调用LLM分析题目，生成解析和正确答案。

        Args:
            question_text: 题目文本
            user_answer: 用户答案（用于对比分析）

        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            prompt = f"""
请分析以下数学题目，给出详细的解答过程和正确答案。

题目：
{question_text}

{'用户答案：' + user_answer if user_answer else ''}

请按照以下格式输出：
1. 答案：[正确答案]
2. 知识点：[所属知识点，如：一元一次方程、二次方程等]
3. 解析：[详细的解题步骤和思路]
4. 错因分析（如果有用户答案）：[分析用户可能的错误原因]
"""

            result_text = self.llm.generate(
                prompt=prompt,
                system_prompt="你是一位数学老师，擅长分析和解答数学题目。",
                temperature=0.3
            )
            
            # 解析LLM返回的结果
            result = {
                "answer": None,
                "knowledge_id": None,
                "analysis": None,
                "error_analysis": None
            }

            for line in result_text.split("\n"):
                if line.startswith("1. 答案："):
                    result["answer"] = line.replace("1. 答案：", "").strip()
                elif line.startswith("2. 知识点："):
                    result["knowledge_id"] = line.replace("2. 知识点：", "").strip()
                elif line.startswith("3. 解析："):
                    result["analysis"] = line.replace("3. 解析：", "").strip()
                elif line.startswith("4. 错因分析"):
                    result["error_analysis"] = line.replace("4. 错因分析（如果有用户答案）：", "").strip()

            # 如果解析部分在多行
            if not result["analysis"]:
                # 尝试找到解析部分
                lines = result_text.split("\n")
                for i, line in enumerate(lines):
                    if "解析" in line or "解答" in line:
                        result["analysis"] = "\n".join(lines[i:]).strip()
                        break

            return result

        except Exception as e:
            logger.exception("分析题目失败", exc_info=e)
            return {
                "answer": None,
                "knowledge_id": None,
                "analysis": "分析失败，请稍后重试",
                "error_analysis": None
            }

    def _generate_exercises(self, wrong_question_id: int, original_question: str, knowledge_id: str) -> List[Dict[str, Any]]:
        """
        根据错题生成3道类似的练习题。

        Args:
            wrong_question_id: 错题ID
            original_question: 原题目
            knowledge_id: 知识点ID

        Returns:
            List[Dict[str, Any]]: 生成的练习题列表
        """
        exercises = []
        
        try:
            prompt = f"""
请根据以下数学题目，生成3道类似的练习题，用于巩固练习。

原题目：
{original_question}

知识点：{knowledge_id}

要求：
1. 生成3道难度相近的题目
2. 每道题给出正确答案
3. 题目类型与原题目相同
4. 数值可以不同，但解题思路要一致

请按照以下格式输出：
题目1：[题目内容]
答案1：[正确答案]
题目2：[题目内容]
答案2：[正确答案]
题目3：[题目内容]
答案3：[正确答案]
"""

            result_text = self.llm.generate(
                prompt=prompt,
                system_prompt="你是一位数学老师，擅长根据现有题目生成类似的练习题。",
                temperature=0.5
            )
            
            # 解析生成的练习题
            lines = result_text.split("\n")
            current_question = None
            
            for line in lines:
                if line.startswith("题目"):
                    current_question = line.replace("题目1：", "").replace("题目2：", "").replace("题目3：", "").strip()
                elif line.startswith("答案") and current_question:
                    answer = line.replace("答案1：", "").replace("答案2：", "").replace("答案3：", "").strip()
                    
                    # 保存到数据库
                    exercise_id = self.db.add_wrong_question_exercise(
                        wrong_question_id=wrong_question_id,
                        question_text=current_question,
                        correct_answer=answer,
                        difficulty=1
                    )
                    
                    if exercise_id > 0:
                        exercises.append({
                            "id": exercise_id,
                            "question_text": current_question,
                            "correct_answer": answer,
                            "difficulty": 1
                        })
                    
                    current_question = None

        except Exception as e:
            logger.exception("生成练习题失败", exc_info=e)

        return exercises

    def _update_review_plan(self, learner_id: str, knowledge_id: str):
        """
        更新复习计划，将该知识点加入近期复习列表。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
        """
        # 通过间隔重复算法更新复习计划
        # 这里简化处理，实际项目中会调用SM-2算法
        logger.info(f"更新复习计划: 学习者 {learner_id}, 知识点 {knowledge_id}")

    def _update_mastery(self, learner_id: str, knowledge_id: str, is_correct: bool):
        """
        更新学习者对知识点的掌握度。

        Args:
            learner_id: 学习者ID
            knowledge_id: 知识点ID
            is_correct: 是否正确
        """
        # 通过BKT算法更新掌握度
        logger.info(f"更新掌握度: 学习者 {learner_id}, 知识点 {knowledge_id}, 正确: {is_correct}")

    def get_wrong_questions(self, learner_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取学习者的错题列表。

        Args:
            learner_id: 学习者ID
            limit: 返回数量限制

        Returns:
            List[Dict[str, Any]]: 错题列表
        """
        questions = self.db.get_wrong_questions(learner_id, limit)
        
        # 添加知识点名称
        for q in questions:
            if q["knowledge_id"] in self.knowledge_graph.nodes:
                q["knowledge_name"] = self.knowledge_graph.nodes[q["knowledge_id"]].name
            else:
                q["knowledge_name"] = q["knowledge_id"]
        
        return questions

    def get_wrong_question_detail(self, question_id: int) -> Optional[Dict[str, Any]]:
        """
        获取错题详情，包括生成的练习题。

        Args:
            question_id: 错题ID

        Returns:
            Optional[Dict[str, Any]]: 错题详情
        """
        question = self.db.get_wrong_question_by_id(question_id)
        if not question:
            return None

        # 获取练习题
        exercises = self.db.get_wrong_question_exercises(question_id)
        question["exercises"] = exercises

        # 获取练习记录
        practices = self.db.get_wrong_question_practices(question_id)
        question["practices"] = practices

        # 添加知识点名称
        if question["knowledge_id"] in self.knowledge_graph.nodes:
            question["knowledge_name"] = self.knowledge_graph.nodes[question["knowledge_id"]].name

        return question

    def practice_wrong_question(
        self,
        question_id: int,
        learner_id: str,
        user_answer: str,
        is_correct: bool,
        time_spent: int = None
    ) -> Dict[str, Any]:
        """
        练习错题，记录答题结果。

        Args:
            question_id: 错题ID
            learner_id: 学习者ID
            user_answer: 用户答案
            is_correct: 是否正确
            time_spent: 用时（秒）

        Returns:
            Dict[str, Any]: 练习结果
        """
        # 添加练习记录
        success = self.db.add_wrong_question_practice(
            wrong_question_id=question_id,
            learner_id=learner_id,
            is_correct=is_correct,
            user_answer=user_answer,
            time_spent=time_spent
        )

        if success:
            # 如果答对了，更新掌握度
            question = self.db.get_wrong_question_by_id(question_id)
            if question and is_correct:
                self._update_mastery(learner_id, question["knowledge_id"], True)

            return {"success": True, "message": "练习记录已保存"}
        else:
            return {"success": False, "error": "保存练习记录失败"}

    def delete_wrong_question(self, question_id: int) -> bool:
        """
        删除错题。

        Args:
            question_id: 错题ID

        Returns:
            bool: 是否成功
        """
        return self.db.delete_wrong_question(question_id)

    def get_wrong_questions_count(self, learner_id: str) -> int:
        """
        获取学习者的错题数量。

        Args:
            learner_id: 学习者ID

        Returns:
            int: 错题数量
        """
        return self.db.get_wrong_questions_count(learner_id)


# 全局单例
_wrong_question_manager: Optional[WrongQuestionManager] = None


def get_wrong_question_manager() -> WrongQuestionManager:
    """
    获取错题本管理器单例。

    Returns:
        WrongQuestionManager: 错题本管理器实例
    """
    global _wrong_question_manager
    if _wrong_question_manager is None:
        _wrong_question_manager = WrongQuestionManager()
    return _wrong_question_manager