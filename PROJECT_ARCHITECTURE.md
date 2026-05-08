# 多Agent智能教育与个性化学习系统 - 项目架构文档

## 1. 项目概述

本项目是一个基于多Agent协作的智能教育系统，旨在通过人工智能技术为学生提供个性化的学习体验。系统集成了知识点评估、智能教学、课程规划、分级提示等功能，利用贝叶斯知识追踪（BKT）、间隔重复算法（SM-2）和知识图谱等核心技术，结合LangGraph进行Agent编排，以及大语言模型（LLM）生成动态教学内容，实现自适应学习。

## 2. 技术栈概览

| 类别 | 技术/框架 | 版本 | 用途 |
|------|-----------|------|------|
| 后端框架 | FastAPI | 0.112.2 | REST API和WebSocket服务 |
| 前端框架 | Streamlit | 1.37.1 | 交互式前端界面 |
| Agent编排 | LangGraph | 0.2.14 | 状态图编排和Agent协作 |
| 大语言模型 | OpenAI API | 1.42.0 | 生成教学内容和提示 |
| 数据库 | SQLAlchemy | 2.0.32 | ORM和数据持久化 |
| 数据验证 | Pydantic | 2.8.2 | 数据模型和验证 |
| 图算法 | NetworkX | 3.3 | 知识图谱构建 |
| 可视化 | PyVis | 0.3.2 | 知识图谱可视化 |
| 测试 | pytest | 8.3.2 | 单元测试和集成测试 |

## 3. 目录结构

```
multi-agent-education/
├── agents/                # Agent实现
│   ├── __init__.py
│   ├── assessment_agent.py # 评估Agent (BKT算法)
│   ├── base_agent.py      # Agent基类
│   ├── curriculum_agent.py # 课程规划Agent
│   ├── engagement_agent.py # 互动Agent
│   ├── hint_agent.py      # 提示Agent
│   └── tutor_agent.py     # 教学Agent
├── api/                   # 接口层
│   ├── __init__.py
│   ├── main.py            # FastAPI入口
│   ├── monitor_utils.py   # 监控工具
│   ├── orchestrator.py    # Agent编排器
│   ├── routes.py          # REST路由
│   └── websocket.py       # WebSocket支持
├── config/                # 配置文件
│   ├── __init__.py
│   └── settings.py        # 全局配置
├── core/                  # 核心模块
│   ├── __init__.py
│   ├── database.py        # 数据库
│   ├── event_bus.py       # 事件总线
│   ├── graph.py           # LangGraph状态图
│   ├── knowledge_graph.py # 知识图谱
│   ├── learner_model.py   # BKT学习者模型
│   ├── learner_model_manager.py # 学习者模型管理
│   ├── llm.py             # LLM客户端
│   ├── observability.py   # 可观察性
│   └── spaced_repetition.py # SM-2间隔重复算法
├── docs/                  # 文档
│   ├── architecture.md
│   ├── deployment.md
│   ├── interview-guide.md
│   └── knowledge-points.md
├── examples/              # 示例图片
│   ├── demo1_chat.png
│   ├── demo2_knowledge.png
│   ├── demo3_practice.png
│   └── demo4_progress.png
├── scripts/               # 脚本文件
│   ├── demo_competition.ps1
│   └── demo_competition.py
├── tests/                 # 测试文件
│   ├── __init__.py
│   ├── test_agents.py
│   └── test_graph.py
├── .env.example           # 环境变量示例
├── .gitignore
├── LICENSE
├── PROJECT_ARCHITECTURE.md # 项目架构文档
├── README.md              # 项目说明
├── requirements.txt       # 依赖
└── streamlit_app.py       # Streamlit前端
```

### 3.1 目录说明

| 目录 | 作用 | 详细说明 |
|------|------|----------|
| **agents/** | Agent实现 | 包含各种功能Agent的实现，负责不同的教育任务 |
| **api/** | 接口层 | 提供REST API和WebSocket服务，处理前端请求 |
| **config/** | 配置文件 | 管理全局配置，从环境变量加载设置 |
| **core/** | 核心模块 | 包含核心算法、数据结构和工具类 |
| **docs/** | 文档 | 项目文档，包括架构、部署指南等 |
| **examples/** | 示例图片 | 项目演示图片 |
| **scripts/** | 脚本文件 | 辅助脚本，如演示脚本 |
| **tests/** | 测试文件 | 单元测试和集成测试 |

## 4. 核心开发规范

### 4.1 代码风格

- 使用Python 3.10+语法
- 遵循PEP 8代码风格
- 使用类型注解提高代码可读性和IDE支持
- 为关键函数添加文档字符串（Google风格）

### 4.2 如何添加新页面

1. **Streamlit前端页面**：
   - 在`streamlit_app.py`中添加新的tab
   - 使用`st.tabs()`函数创建新的标签页
   - 实现页面逻辑和UI组件
   - 参考现有的tab实现方式

2. **后端API端点**：
   - 在`api/routes.py`中添加新的路由
   - 使用FastAPI的装饰器定义路由
   - 创建对应的Pydantic模型用于请求和响应
   - 实现处理逻辑，调用Orchestrator的方法

### 4.3 如何调用接口

1. **前端直接调用**：
   - 实例化`AgentOrchestrator`对象
   - 直接调用其方法，如`ask_question()`、`submit_answer()`等
   - 处理返回的事件列表，提取所需信息

2. **HTTP请求**：
   - 使用标准HTTP客户端（如`urllib`、`requests`）
   - 访问API端点，如`/api/submit`、`/api/question`等
   - 发送JSON格式的请求数据
   - 处理JSON格式的响应数据

### 4.4 如何添加新Agent

1. **创建Agent类**：
   - 继承自`BaseAgent`类
   - 实现必要的方法，如`process()`
   - 注册事件处理函数

2. **注册到Orchestrator**：
   - 在`api/orchestrator.py`中的`AgentOrchestrator`类中添加新Agent的实例
   - 确保在`__init__`方法中正确初始化

### 4.5 如何添加新的知识点

1. **修改知识图谱**：
   - 在`core/knowledge_graph.py`中修改`build_sample_math_graph()`函数
   - 添加新的`KnowledgeNode`对象
   - 定义知识点之间的依赖关系

2. **更新前端选项**：
   - 前端会自动从知识图谱中获取知识点列表
   - 无需手动更新前端代码

## 5. 核心模块说明

### 5.1 Agent系统

- **BaseAgent**：所有Agent的基类，提供事件订阅和发布功能
- **AssessmentAgent**：使用BKT算法评估学生对知识点的掌握程度
- **TutorAgent**：生成苏格拉底式教学回复，引导学生思考
- **CurriculumAgent**：根据学生掌握情况推荐下一个知识点和复习计划
- **HintAgent**：学生遇到困难时提供分级提示
- **EngagementAgent**：监测学生互动情况，提供鼓励和支持

### 5.2 核心算法

- **BKT（贝叶斯知识追踪）**：实时追踪学生对知识点的掌握程度
- **SM-2（间隔重复算法）**：优化复习间隔，提高记忆效率
- **知识图谱**：管理知识点之间的依赖关系，支持学习路径规划

### 5.3 状态管理

- **LangGraph状态图**：使用StateGraph实现状态管理和条件路由
- **LearningState**：定义学习过程中的状态数据结构
- **MemorySaver**：保存对话上下文，支持多轮对话

### 5.4 记忆系统

- **短期记忆**：使用LangGraph的MemorySaver保存对话上下文
- **长期记忆**：将学习者模型和学习历史持久化到SQLite数据库

## 6. 接口说明

### 6.1 REST API

| 端点 | 方法 | 功能 | 请求体 | 响应 |
|------|------|------|--------|------|
| `/api/submit` | POST | 提交答题结果 | `{"learner_id": "...", "knowledge_id": "...", "is_correct": true, "time_spent_seconds": 30, "error_type": "concept"}` | `{"learner_id": "...", "events": [...]}` |
| `/api/question` | POST | 学生提问 | `{"learner_id": "...", "knowledge_id": "...", "question": "..."}` | `{"learner_id": "...", "events": [...]}` |
| `/api/message` | POST | 发送自由消息 | `{"learner_id": "...", "message": "...", "knowledge_id": "..."}` | `{"learner_id": "...", "events": [...]}` |
| `/api/progress/{learner_id}` | GET | 获取学习进度 | N/A | 学习进度数据 |
| `/api/review-plan/{learner_id}` | GET | 获取复习计划 | N/A | 复习计划数据 |
| `/api/monitor/summary` | GET | 获取监控汇总 | N/A | 监控数据 |
| `/api/health` | GET | 健康检查 | N/A | `{"status": "ok"}` |

### 6.2 WebSocket

- **WebSocket端点**：`/ws`
- **功能**：实时交互，支持聊天和通知
- **消息格式**：JSON格式，包含事件类型和数据

## 7. 部署与运行

### 7.1 环境要求

- Python 3.10+
- pip 或 conda
- 可选：Docker

### 7.2 安装步骤

1. 克隆项目：
   ```bash
   git clone https://github.com/your-username/multi-agent-education.git
   cd multi-agent-education
   ```

2. 创建虚拟环境：
   ```bash
   python -m venv venv
   venv\Scripts\activate     # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 配置环境变量：
   复制`.env.example`到`.env`并填写配置

5. 运行后端：
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. 运行前端：
   ```bash
   streamlit run streamlit_app.py
   ```

### 7.3 访问地址

- 前端界面：http://localhost:8501
- API文档：http://localhost:8000/docs

## 8. 开发流程

1. **需求分析**：明确功能需求和技术要求
2. **设计**：设计数据结构、API接口和UI界面
3. **实现**：编写代码，实现功能
4. **测试**：运行测试，确保功能正常
5. **部署**：部署到生产环境
6. **监控**：监控系统运行状态

## 9. 最佳实践

1. **代码组织**：
   - 遵循模块化设计，将功能分离到不同的模块
   - 使用类型注解提高代码可读性
   - 为关键函数添加文档字符串

2. **错误处理**：
   - 捕获并处理异常，避免系统崩溃
   - 记录错误日志，便于调试
   - 向用户提供友好的错误信息

3. **性能优化**：
   - 优化数据库查询，避免重复查询
   - 缓存频繁使用的数据
   - 合理使用异步编程，提高并发性能

4. **安全性**：
   - 验证用户输入，防止注入攻击
   - 保护敏感信息，避免泄露
   - 实现适当的访问控制

## 10. 故障排查

### 10.1 常见问题

1. **LLM调用失败**：
   - 检查API密钥是否正确
   - 检查网络连接是否正常
   - 检查模型名称是否正确

2. **数据库连接失败**：
   - 检查数据库配置是否正确
   - 检查数据库服务是否运行

3. **前端加载失败**：
   - 检查Streamlit服务是否运行
   - 检查浏览器是否支持

### 10.2 调试技巧

1. **日志查看**：
   - 查看FastAPI日志，了解后端运行状态
   - 查看Streamlit日志，了解前端运行状态

2. **断点调试**：
   - 使用IDE的断点调试功能，逐步执行代码
   - 检查变量值，定位问题所在

3. **API测试**：
   - 使用FastAPI的自动生成的API文档测试接口
   - 使用curl或Postman测试API接口

## 11. 未来扩展

1. **支持更多学科**：
   - 扩展知识图谱，支持更多学科的知识点
   - 调整算法参数，适应不同学科的特点

2. **新增功能**：
   - 错题本功能，记录学生的错题
   - 代码执行环境，支持编程题
   - 教师后台，支持班级管理

3. **技术升级**：
   - 支持更多LLM模型
   - 优化LangGraph状态图，提高性能
   - 实现更复杂的学习路径推荐算法

## 12. 总结

本项目是一个完整的智能教育系统，通过多Agent协作和先进的教育算法，为学生提供个性化的学习体验。系统集成了知识点评估、智能教学、课程规划、分级提示等功能，利用贝叶斯知识追踪、间隔重复算法和知识图谱等核心技术，结合大语言模型生成动态教学内容，实现自适应学习。

项目采用了现代化的技术栈和架构设计，具有良好的可扩展性和可维护性。通过本架构文档，开发者可以快速了解项目的全局规范和架构，为后续的开发和维护提供指导。