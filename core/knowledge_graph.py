"""
知识图谱 -- 管理知识点之间的依赖关系。

知识图谱是一个 DAG（有向无环图），用于：
1. 确定学习顺序（拓扑排序）
2. 检查前置知识是否达标
3. 推荐下一个可学习的知识点

面试要点：
- DAG拓扑排序算法（Kahn's / DFS）
- 前置知识检查的重要性
- 知识图谱 vs 知识树：图更灵活，一个节点可有多个前置
"""

import logging
from collections import deque
from typing import Dict, List, Set, Optional

from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeNode(BaseModel):
    """知识点节点模型。"""
    id: str
    name: str
    subject: str = "math"  # 学科
    difficulty: float = 0.5  # 难度 0-1（0最简单，1最难）
    description: str = ""
    prerequisites: List[str] = Field(default_factory=list)  # 前置知识点ID列表
    tags: List[str] = Field(default_factory=list)  # 知识点标签


class KnowledgeGraph:
    """
    知识图谱（DAG），管理知识点依赖关系。

    核心特性：
    - 维护知识点的前置/后继关系
    - 提供拓扑排序能力（确定学习顺序）
    - 推荐可学习的知识点
    - 生成学习路径
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, KnowledgeNode] = {}  # 节点ID -> 节点对象
        self._adjacency: Dict[str, List[str]] = {}  # 邻接表：节点ID -> 后继节点列表
        self._reverse_adj: Dict[str, List[str]] = {}  # 反向邻接表：节点ID -> 前置节点列表

    def add_node(self, node: KnowledgeNode) -> None:
        """
        添加知识点节点，并维护邻接表。

        Args:
            node: 知识点节点对象
        """
        # 添加节点
        self.nodes[node.id] = node

        # 初始化邻接表
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []
        if node.id not in self._reverse_adj:
            self._reverse_adj[node.id] = []

        # 维护前置/后继关系
        for prereq_id in node.prerequisites:
            # 确保前置节点存在于邻接表中
            if prereq_id not in self._adjacency:
                self._adjacency[prereq_id] = []
            if prereq_id not in self._reverse_adj:
                self._reverse_adj[prereq_id] = []

            # 更新邻接表：前置节点 -> 当前节点（后继）
            self._adjacency[prereq_id].append(node.id)
            # 更新反向邻接表：当前节点 -> 前置节点
            self._reverse_adj[node.id].append(prereq_id)

        logger.debug(f"Added node {node.id} with prerequisites: {node.prerequisites}")

    def get_prerequisites(self, node_id: str) -> List[str]:
        """获取节点的直接前置知识点ID列表。"""
        return self._reverse_adj.get(node_id, [])

    def get_successors(self, node_id: str) -> List[str]:
        """获取节点的直接后继知识点ID列表。"""
        return self._adjacency.get(node_id, [])

    def get_all_prerequisites(self, node_id: str) -> Set[str]:
        """
        递归获取节点的所有前置知识点（包括间接前置）。

        Args:
            node_id: 目标节点ID

        Returns:
            所有前置知识点ID的集合
        """
        visited: Set[str] = set()
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found in graph")
            return visited

        # BFS遍历所有前置节点
        queue = deque(self.get_prerequisites(node_id))
        while queue:
            pid = queue.popleft()
            if pid not in visited:
                visited.add(pid)
                queue.extend(self.get_prerequisites(pid))

        return visited

    def topological_sort(self) -> List[str]:
        """
        拓扑排序（Kahn算法）- 确定知识点的学习顺序。

        Returns:
            按学习顺序排列的知识点ID列表
        """
        # 计算入度
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for nid in self.nodes:
            for succ in self._adjacency.get(nid, []):
                if succ in in_degree:
                    in_degree[succ] += 1

        # 初始化队列（入度为0的节点）
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        result: List[str] = []

        # BFS处理
        while queue:
            nid = queue.popleft()
            result.append(nid)
            for succ in self._adjacency.get(nid, []):
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        # 检查是否有环
        if len(result) != len(self.nodes):
            logger.warning(f"Knowledge graph contains a cycle! Found {len(result)} nodes out of {len(self.nodes)}")

        logger.info(f"Topological sort completed with {len(result)} nodes")
        return result

    def get_ready_nodes(self, mastered_ids: Set[str]) -> List[str]:
        """
        获取当前可学习的知识点（前置知识全部掌握，但自身未掌握）。

        Args:
            mastered_ids: 已掌握的知识点ID集合

        Returns:
            按难度排序的可学习知识点ID列表
        """
        ready_nodes = []
        for nid, node in self.nodes.items():
            # 跳过已掌握的节点
            if nid in mastered_ids:
                continue

            # 检查前置知识是否全部掌握
            prereqs = set(node.prerequisites)
            if prereqs.issubset(mastered_ids):
                ready_nodes.append(nid)

        # 按难度升序排列（先学简单的）
        ready_nodes.sort(key=lambda nid: self.nodes[nid].difficulty)
        logger.debug(f"Ready nodes for mastered {mastered_ids}: {ready_nodes}")
        return ready_nodes

    def get_learning_path(self, target_id: str, mastered_ids: Set[str]) -> List[str]:
        """
        生成到达目标知识点的最短学习路径。

        Args:
            target_id: 目标知识点ID
            mastered_ids: 已掌握的知识点ID集合

        Returns:
            按学习顺序排列的知识点ID列表
        """
        if target_id not in self.nodes:
            logger.warning(f"Target node {target_id} not found")
            return []

        # 获取所有需要学习的前置知识
        needed_prereqs = self.get_all_prerequisites(target_id) - mastered_ids
        # 包含目标节点本身（如果未掌握）
        if target_id not in mastered_ids:
            needed_prereqs.add(target_id)

        # 按拓扑序筛选需要学习的节点
        full_order = self.topological_sort()
        learning_path = [nid for nid in full_order if nid in needed_prereqs]

        logger.info(f"Learning path to {target_id}: {learning_path}")
        return learning_path

    def get_node_by_id(self, node_id: str) -> Optional[KnowledgeNode]:
        """按ID获取知识点节点对象。"""
        return self.nodes.get(node_id)

    def get_nodes_by_tag(self, tag: str) -> List[KnowledgeNode]:
        """按标签筛选知识点节点。"""
        return [node for node in self.nodes.values() if tag in node.tags]

    def get_nodes_by_subject(self, subject: str) -> List[KnowledgeNode]:
        """按学科筛选知识点节点。"""
        return [node for node in self.nodes.values() if node.subject == subject]


def build_sample_math_graph() -> KnowledgeGraph:
    """构建示例数学知识图谱（初中数学核心知识点）。"""
    graph = KnowledgeGraph()

    # 定义知识点节点
    nodes = [
        # 基础运算
        KnowledgeNode(
            id="arithmetic",
            name="四则运算",
            difficulty=0.1,
            description="加减乘除基本运算规则和技巧",
            tags=["基础", "运算"]
        ),
        KnowledgeNode(
            id="fractions",
            name="分数运算",
            difficulty=0.2,
            prerequisites=["arithmetic"],
            description="分数的加减乘除、约分、通分等运算",
            tags=["基础", "运算"]
        ),
        KnowledgeNode(
            id="negative_numbers",
            name="负数",
            difficulty=0.15,
            prerequisites=["arithmetic"],
            description="负数的概念、大小比较和运算规则",
            tags=["基础", "数论"]
        ),
        # 代数基础
        KnowledgeNode(
            id="algebraic_expr",
            name="代数式",
            difficulty=0.3,
            prerequisites=["arithmetic", "negative_numbers"],
            description="用字母表示数、代数式的化简和求值",
            tags=["代数", "基础"]
        ),
        # 方程系列
        KnowledgeNode(
            id="linear_eq_1",
            name="一元一次方程",
            difficulty=0.35,
            prerequisites=["algebraic_expr"],
            description="一元一次方程的解法和应用",
            tags=["代数", "方程"]
        ),
        KnowledgeNode(
            id="linear_eq_2",
            name="二元一次方程组",
            difficulty=0.45,
            prerequisites=["linear_eq_1"],
            description="二元一次方程组的代入消元、加减消元解法",
            tags=["代数", "方程"]
        ),
        KnowledgeNode(
            id="factoring",
            name="因式分解",
            difficulty=0.4,
            prerequisites=["algebraic_expr"],
            description="提取公因式、公式法、十字相乘法等因式分解方法",
            tags=["代数", "多项式"]
        ),
        KnowledgeNode(
            id="quadratic_eq",
            name="一元二次方程",
            difficulty=0.55,
            prerequisites=["factoring", "linear_eq_1"],
            description="一元二次方程的配方法、公式法、因式分解法求解",
            tags=["代数", "方程"]
        ),
        # 函数系列
        KnowledgeNode(
            id="quadratic_func",
            name="二次函数",
            difficulty=0.6,
            prerequisites=["quadratic_eq"],
            description="二次函数的图像、性质和应用",
            tags=["代数", "函数"]
        ),
        KnowledgeNode(
            id="inequality",
            name="不等式",
            difficulty=0.4,
            prerequisites=["linear_eq_1"],
            description="一元一次不等式（组）的解法和应用",
            tags=["代数", "不等式"]
        ),
        KnowledgeNode(
            id="coordinate",
            name="平面直角坐标系",
            difficulty=0.3,
            prerequisites=["negative_numbers"],
            description="坐标系的概念、点的坐标、距离计算",
            tags=["几何", "坐标"]
        ),
        KnowledgeNode(
            id="linear_func",
            name="一次函数",
            difficulty=0.45,
            prerequisites=["linear_eq_1", "coordinate"],
            description="一次函数的图像、性质、解析式求解",
            tags=["代数", "函数"]
        ),
        # 几何系列
        KnowledgeNode(
            id="pythagorean",
            name="勾股定理",
            difficulty=0.35,
            prerequisites=["arithmetic"],
            description="直角三角形三边关系及应用",
            tags=["几何", "三角形"]
        ),
        KnowledgeNode(
            id="similar_triangle",
            name="相似三角形",
            difficulty=0.5,
            prerequisites=["pythagorean", "fractions"],
            description="相似三角形的判定和性质",
            tags=["几何", "三角形"]
        ),
        KnowledgeNode(
            id="trig_basic",
            name="三角函数基础",
            difficulty=0.55,
            prerequisites=["pythagorean", "fractions"],
            description="正弦、余弦、正切的定义和简单应用",
            tags=["几何", "三角"]
        ),
        # 统计概率
        KnowledgeNode(
            id="probability",
            name="概率初步",
            difficulty=0.4,
            prerequisites=["fractions"],
            description="随机事件、概率计算、古典概型",
            tags=["统计", "概率"]
        ),
        KnowledgeNode(
            id="statistics",
            name="数据统计",
            difficulty=0.35,
            prerequisites=["arithmetic", "fractions"],
            description="平均数、中位数、众数、方差等统计量计算",
            tags=["统计", "数据"]
        ),
        # 其他代数
        KnowledgeNode(
            id="sequence",
            name="数列",
            difficulty=0.5,
            prerequisites=["algebraic_expr"],
            description="等差数列、等比数列的通项和求和",
            tags=["代数", "数列"]
        ),
        KnowledgeNode(
            id="sets",
            name="集合",
            difficulty=0.25,
            prerequisites=["arithmetic"],
            description="集合的概念、表示方法、基本运算",
            tags=["基础", "数论"]
        ),
        KnowledgeNode(
            id="logic",
            name="简易逻辑",
            difficulty=0.3,
            prerequisites=["sets"],
            description="命题、充要条件、逻辑联结词",
            tags=["基础", "逻辑"]
        ),
    ]

    # 添加所有节点到图谱
    for node in nodes:
        graph.add_node(node)

    logger.info(f"Sample math knowledge graph built with {len(nodes)} nodes")
    return graph


# 测试代码
if __name__ == "__main__":
    # 构建测试图谱
    graph = build_sample_math_graph()

    # 测试拓扑排序
    print("拓扑排序结果：")
    for i, nid in enumerate(graph.topological_sort()):
        print(f"{i+1}. {graph.nodes[nid].name}")

    # 测试可学习节点
    mastered = {"arithmetic", "negative_numbers"}
    print(f"\n已掌握：{[graph.nodes[nid].name for nid in mastered]}")
    print("可学习的知识点：")
    for nid in graph.get_ready_nodes(mastered):
        print(f"- {graph.nodes[nid].name} (难度：{graph.nodes[nid].difficulty})")

    # 测试学习路径
    target = "quadratic_eq"
    print(f"\n到达 {graph.nodes[target].name} 的学习路径：")
    for i, nid in enumerate(graph.get_learning_path(target, mastered)):
        print(f"{i+1}. {graph.nodes[nid].name}")