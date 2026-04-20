"""
多Agent智能学习系统 - Streamlit 交互式前端。
功能：
1. 学生登录
2. 知识点选择
3. 答题界面
4. 实时聊天
5. 学习进度展示
6. 知识图谱可视化
7. 系统监控（竞赛演示：延迟、错误率、掌握度分布、Agent 漏斗）
"""
import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import urllib.error
import urllib.request
from urllib.parse import quote
from datetime import datetime

# 导入核心模块
from api.monitor_utils import build_agent_event_funnel
from api.orchestrator import AgentOrchestrator
from core.knowledge_graph import build_sample_math_graph, KnowledgeGraph, KnowledgeNode
from core.database import get_database
from core.llm import get_llm_client
from core.observability import get_http_metrics_snapshot

# 页面配置
st.set_page_config(
    page_title="多Agent智能学习系统",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 移动端适配：隐藏右上角菜单与底部水印 + 输入控件尽量占满宽度
st.markdown(
    "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} "
    ".stTextInput input, .stTextArea textarea {width: 100% !important;}</style>",
    unsafe_allow_html=True,
)

# 初始化Session State
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "learner_id" not in st.session_state:
    st.session_state.learner_id = "student_001"
if "current_knowledge" not in st.session_state:
    st.session_state.current_knowledge = "arithmetic"
if "show_progress" not in st.session_state:
    st.session_state.show_progress = False
if "generated_question" not in st.session_state:
    st.session_state.generated_question = ""

# 侧边栏：学生信息与设置
with st.sidebar:
    st.title("🧠 多Agent智能学习系统")
    st.divider()

    # 学生ID输入
    st.session_state.learner_id = st.text_input(
        "👤 学生ID",
        value=st.session_state.learner_id,
        help="输入任意学生ID，系统会自动保存学习进度"
    )

    # 构建知识图谱（确保全局可用）
    knowledge_graph = build_sample_math_graph()
    knowledge_options = list(knowledge_graph.nodes.keys())

    # 知识点选择
    st.session_state.current_knowledge = st.selectbox(
        "📚 选择知识点",
        options=knowledge_options,
        index=knowledge_options.index(
            st.session_state.current_knowledge) if st.session_state.current_knowledge in knowledge_options else 0,
        help="选择你想学习的知识点"
    )

    # 显示当前知识点信息
    if st.session_state.current_knowledge in knowledge_graph.nodes:
        node = knowledge_graph.nodes[st.session_state.current_knowledge]
        st.info(f"**{node.name}**\n\n难度：{'⭐' * int(node.difficulty * 5)}\n\n{node.description}")

    st.divider()

    # 学习进度快速查看
    if st.button("📊 查看我的学习进度", use_container_width=True):
        st.session_state.show_progress = True

    # 清空聊天记录
    if st.button("🗑️ 清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.success("聊天记录已清空！")

# 主界面
tab1, tab2, tab3, tab4 = st.tabs(["💬 学习聊天", "📝 答题练习", "📊 学习进度", "📡 系统监控"])

# --- Tab 1: 学习聊天 ---
with tab1:
    st.header(f"学习聊天 - {st.session_state.current_knowledge}")

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 聊天输入
    if prompt := st.chat_input("问我任何问题，或者告诉我你在学习中遇到的困难..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用Agent处理
        with st.chat_message("assistant"):
            with st.spinner("AI老师正在思考..."):
                try:
                    # 运行异步函数
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    events = loop.run_until_complete(
                        st.session_state.orchestrator.ask_question(
                            st.session_state.learner_id,
                            st.session_state.current_knowledge,
                            prompt,
                            chat_history=st.session_state.messages[-20:]
                        )
                    )

                    # 提取教学回复
                    response = "抱歉，我现在无法回答你的问题。"
                    for event in events:
                        if event.get("type") == "tutor.teaching_response":
                            response = event.get("data", {}).get("response", response)

                    st.markdown(response)

                    # 添加助手消息
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"发生错误：{str(e)}")

# --- Tab 2: 答题练习 ---
with tab2:
    st.header(f"答题练习 - {st.session_state.current_knowledge}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("题目")
        question_type = st.radio(
            "题型",
            ("自动", "选择", "填空", "解答"),
            horizontal=True,
        )

        # 自动出题按钮：替代手动输入（仍保留可编辑，以便二次微调）
        if st.button("✨ 自动出题", type="primary", use_container_width=True):
            with st.spinner("正在自动出题..."):
                try:
                    learner_model = st.session_state.orchestrator.learner_model_manager.get_model(
                        st.session_state.learner_id
                    )
                    mastery = 0.1
                    if learner_model:
                        try:
                            mastery = learner_model.get_state(st.session_state.current_knowledge).mastery
                        except Exception:
                            mastery = 0.1

                    kg_node = knowledge_graph.nodes.get(st.session_state.current_knowledge)
                    knowledge_point = (
                        f"{kg_node.name}（{st.session_state.current_knowledge}）" if kg_node else st.session_state.current_knowledge
                    )

                    llm = get_llm_client()
                    st.session_state.generated_question = llm.generate_question(
                        knowledge_point=knowledge_point,
                        mastery=mastery,
                        question_type=question_type,  # type: ignore[arg-type]
                    )
                except Exception as e:
                    st.error(f"自动出题失败：{str(e)}")

        question_text = st.text_area(
            "题目内容（可编辑）",
            height=160,
            value=st.session_state.generated_question,
            placeholder="点击上方「自动出题」生成题目，或在此手动编辑"
        )

        st.subheader("你的答案")
        user_answer = st.text_input("输入你的答案")

        is_correct = st.radio(
            "这道题你答对了吗？",
            ("是的，我答对了", "不，我答错了"),
            index=1
        )

        error_type = "unknown"
        if is_correct == "不，我答错了":
            error_type = st.selectbox(
                "错因（用于 SM-2 质量估计）",
                ("concept", "careless", "unknown"),
                index=0,
                format_func=lambda x: {
                    "concept": "概念不清",
                    "careless": "粗心失误",
                    "unknown": "不确定",
                }[x],
                help="概念错误与粗心会影响复习间隔的计算",
            )

        time_spent = st.number_input(
            "花费时间（秒）",
            min_value=0,
            value=30
        )

        if st.button("🚀 提交答案", type="primary", use_container_width=True):
            if not question_text:
                st.warning("请先输入题目内容！")
            else:
                with st.spinner("AI老师正在批改并分析你的掌握情况..."):
                    try:
                        # 运行异步函数
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        events = loop.run_until_complete(
                            st.session_state.orchestrator.submit_answer(
                                st.session_state.learner_id,
                                st.session_state.current_knowledge,
                                is_correct == "是的，我答对了",
                                time_spent,
                                question_text=question_text,
                                answer_text=user_answer,
                                error_type=error_type if is_correct == "不，我答错了" else None,
                            )
                        )

                        # 显示结果
                        st.success("提交成功！")

                        # 提取并显示关键信息
                        mastery = 0.0
                        response = ""

                        for event in events:
                            if event.get("type") == "assessment.mastery_updated":
                                mastery = event.get("data", {}).get("mastery", 0.0)
                            if event.get("type") == "tutor.teaching_response":
                                response = event.get("data", {}).get("response", "")

                        # 显示掌握度
                        st.metric(
                            label=f"{knowledge_graph.nodes[st.session_state.current_knowledge].name} 掌握度",
                            value=f"{mastery:.0%}",
                            delta=f"{(mastery - 0.1):+.0%}" if mastery > 0.1 else None
                        )

                        # 显示AI回复
                        if response:
                            st.divider()
                            st.subheader("AI老师的反馈")
                            st.markdown(response)

                    except Exception as e:
                        st.error(f"发生错误：{str(e)}")

    with col2:
        st.subheader("知识图谱")
        try:
            from streamlit_agraph import agraph, Node, Edge, Config

            # 构建节点和边
            nodes = []
            edges = []
            learner_model = st.session_state.orchestrator.learner_model_manager.get_model(st.session_state.learner_id)

            # 添加所有知识点节点
            for kid, node_data in knowledge_graph.nodes.items():
                # 根据掌握度设置节点颜色
                color = "#97C2FC"  # 默认蓝色
                if learner_model:
                    try:
                        state = learner_model.get_state(kid)
                        mastery = state.mastery
                        if mastery >= 0.85:
                            color = "#00FF00"  # 绿色 - 已掌握
                        elif mastery >= 0.6:
                            color = "#FFFF00"  # 黄色 - 熟练
                        elif mastery >= 0.3:
                            color = "#FFA500"  # 橙色 - 发展中
                        else:
                            color = "#FF0000"  # 红色 - 未掌握
                    except:
                        pass

                nodes.append(Node(
                    id=kid,
                    label=node_data.name,
                    size=25,
                    color=color
                ))

            # 添加知识点之间的边（依赖关系）
            for nid in knowledge_graph.nodes:
                for successor in knowledge_graph.get_successors(nid):
                    edges.append(Edge(
                        source=nid,
                        target=successor,
                        type="CURVE_SMOOTH"
                    ))

            # 配置图谱显示
            config = Config(
                width=300,
                height=300,
                directed=True,
                physics=True,
                hierarchical=False
            )

            # 渲染图谱
            agraph(nodes=nodes, edges=edges, config=config)

            # 显示图例
            st.caption("🟢 已掌握 | 🟡 熟练 | 🟠 发展中 | 🔴 未掌握")

        except ImportError:
            st.error("缺少依赖 streamlit-agraph，请执行：pip install streamlit-agraph")
        except Exception as e:
            st.warning(f"知识图谱加载失败：{str(e)}")
            st.info("不影响学习聊天和答题功能")

# --- Tab 3: 学习进度 ---
with tab3:
    st.header("我的学习进度")

    with st.expander("📅 SM-2 复习计划（间隔重复）", expanded=True):
        st.caption("提交答题后会根据掌握度、耗时与错因更新复习间隔；到期知识点会出现在「今日到期」。")
        plan = st.session_state.orchestrator.get_review_plan(st.session_state.learner_id)
        if plan.get("item_count", 0) == 0:
            st.info("暂无复习记录：请先完成几次「答题练习」，系统将自动生成 SM-2 日程。")
        else:
            due = plan.get("due") or []
            if due:
                st.warning(f"今日/已到期需复习：**{len(due)}** 个知识点")
                st.dataframe(
                    pd.DataFrame(due)[
                        ["name", "next_review", "overdue_days", "interval_days", "easiness_factor", "repetition"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("当前没有已逾期的复习项，保持节奏即可。")

            sched = plan.get("weekly_schedule") or {}
            if sched:
                st.subheader("未来 7 天复习日程")
                sched_rows = [{"日期": d, "知识点数": len(ids), "知识点": "、".join(ids)} for d, ids in sorted(sched.items())]
                st.dataframe(pd.DataFrame(sched_rows), use_container_width=True, hide_index=True)

            up = plan.get("upcoming") or []
            if up:
                st.subheader("即将复习（按时间排序）")
                log_cols = [c for c in ["name", "next_review", "interval_days", "easiness_factor", "repetition", "is_due"] if c in pd.DataFrame(up).columns]
                st.dataframe(pd.DataFrame(up)[log_cols], use_container_width=True, hide_index=True)

            with st.expander("到期后练习反馈（验证间隔重复效果）"):
                for row in up[:8]:
                    logs = row.get("due_cycle_log") or []
                    if logs:
                        st.markdown(f"**{row.get('name')}**")
                        st.json(logs[-5:])

    # 获取学习者进度
    progress = st.session_state.orchestrator.get_learner_progress(st.session_state.learner_id)

    if progress.get("status") == "no_data":
        st.info("你还没有开始学习，去答题或提问吧！")
    else:
        # 总体统计
        col1, col2, col3, col4 = st.columns(4)

        overall = progress.get("progress", {})
        with col1:
            st.metric("总知识点数", overall.get("total_knowledge_points", 0))
        with col2:
            st.metric("平均掌握度", f"{overall.get('avg_mastery', 0):.0%}")
        with col3:
            st.metric("总答题数", overall.get("total_attempts", 0))
        with col4:
            st.metric("正确率", f"{overall.get('accuracy', 0):.0%}")

        st.divider()

        # 知识点掌握情况
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💪 已掌握的知识点")
            strong_points = progress.get("strong_points", [])
            if strong_points:
                for sp in strong_points:
                    node_name = knowledge_graph.nodes.get(sp['id'], sp['id']).name if sp['id'] in knowledge_graph.nodes else sp['id']
                    st.success(f"✅ {node_name} - {sp['mastery']:.0%}")
            else:
                st.info("还没有已掌握的知识点，继续加油！")

        with col2:
            st.subheader("📚 需要加强的知识点")
            weak_points = progress.get("weak_points", [])
            if weak_points:
                for wp in weak_points:
                    node_name = knowledge_graph.nodes.get(wp['id'], wp['id']).name if wp['id'] in knowledge_graph.nodes else wp['id']
                    st.warning(f"⚠️ {node_name} - {wp['mastery']:.0%}")
            else:
                st.success("太棒了！没有需要加强的知识点！")

        st.divider()

        # 学习历史
        st.subheader("📜 最近学习历史")
        db = get_database()
        history = db.get_learning_history(st.session_state.learner_id, limit=20)

        if history:
            history_df = pd.DataFrame(history)
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
            history_df = history_df.sort_values("timestamp", ascending=False)

            # 替换知识点ID为名称
            def get_knowledge_name(kid):
                return knowledge_graph.nodes.get(kid, kid).name if kid in knowledge_graph.nodes else kid

            history_df["knowledge_name"] = history_df["knowledge_id"].apply(get_knowledge_name)

            # 显示表格
            st.dataframe(
                history_df[["timestamp", "knowledge_name", "event_type"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("时间"),
                    "knowledge_name": st.column_config.TextColumn("知识点"),
                    "event_type": st.column_config.TextColumn("事件类型")
                }
            )

            # 学习趋势图
            st.subheader("📈 学习趋势")
            if "mastery" in history_df.columns:
                trend_df = history_df.groupby(["timestamp", "knowledge_name"])["mastery"].mean().reset_index()
                fig = px.line(trend_df, x="timestamp", y="mastery", color="knowledge_name",
                              title="知识点掌握度变化", labels={"mastery": "掌握度", "timestamp": "时间"})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("还没有学习历史记录")

# --- Tab 4: 系统监控（竞赛可展示）---
with tab4:
    st.header("📡 系统监控")
    st.caption(
        "展示 HTTP 接口延迟与错误率、LLM 调用成功率、当前学习者的掌握度分布，以及基于事件总线的多 Agent 协作「漏斗」视图。"
    )
    st.info("请先启动 FastAPI（`uvicorn api.main:app --port 8000`），再点击「刷新监控数据」拉取 `/api/v1/monitor/summary`；未启动时仍可查看本页面内嵌进程的事件与掌握度。")

    api_base = st.text_input(
        "监控 API 根地址（需先启动 FastAPI，默认 8000 端口）",
        value="http://127.0.0.1:8000",
        help="仅当后端运行时，HTTP 延迟等指标才有数据；离线模式下仍可看本进程 EventBus 与掌握度。",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        refresh = st.button("🔄 刷新监控数据", use_container_width=True)
    with col_b:
        show_raw = st.checkbox("显示原始 JSON", value=False)

    if "monitor_cache" not in st.session_state:
        st.session_state.monitor_cache = None

    summary = st.session_state.monitor_cache
    fetch_error = None
    if refresh:
        url = f"{api_base.rstrip('/')}/api/v1/monitor/summary?learner_id={quote(st.session_state.learner_id)}"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                summary = json.loads(resp.read().decode("utf-8"))
            st.session_state.monitor_cache = summary
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            fetch_error = str(e)
            st.session_state.monitor_cache = None
        summary = st.session_state.monitor_cache

    if fetch_error:
        st.warning(f"无法连接监控 API（将使用本进程离线数据）：{fetch_error}")

    metrics = (summary or {}).get("metrics") if summary else get_http_metrics_snapshot()
    if not metrics:
        metrics = get_http_metrics_snapshot()
    http_m = metrics.get("http", {})
    llm_m = metrics.get("llm", {})
    by_path = metrics.get("by_path") or {}

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("HTTP 请求数", http_m.get("total_requests", 0))
    with m2:
        st.metric("HTTP 错误率", f"{(http_m.get('error_rate') or 0) * 100:.2f}%")
    with m3:
        st.metric("P95 延迟(ms)", f"{http_m.get('p95_latency_ms') or 0:.1f}")
    with m4:
        st.metric("LLM 成功率", f"{(llm_m.get('success_rate') or 0) * 100:.1f}%")

    if by_path:
        path_df = pd.DataFrame(
            [{"path": k, **v} for k, v in by_path.items()]
        ).sort_values("avg_latency_ms", ascending=False).head(12)
        st.subheader("接口延迟（Top 路径）")
        fig_lat = px.bar(
            path_df,
            x="path",
            y="avg_latency_ms",
            title="各路径平均延迟（ms）",
            labels={"path": "路径", "avg_latency_ms": "平均延迟(ms)"},
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    local_bus = st.session_state.orchestrator.get_event_bus_stats()
    bus = (summary or {}).get("event_bus") if summary else {
        "total_published": local_bus.get("total_published"),
        "total_handled": local_bus.get("total_handled"),
        "dead_letter_count": local_bus.get("dead_letter_count"),
    }
    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("事件发布", bus.get("total_published") or 0)
    with b2:
        st.metric("事件处理", bus.get("total_handled") or 0)
    with b3:
        st.metric("死信", bus.get("dead_letter_count") or 0)

    if summary and summary.get("agent_funnel"):
        funnel_rows = summary["agent_funnel"]
    else:
        by_type = local_bus.get("by_type") or {}
        funnel_rows = build_agent_event_funnel(by_type if isinstance(by_type, dict) else {})
    funnel_df = pd.DataFrame([{"阶段": r["stage"], "事件量": r["count"]} for r in funnel_rows])
    st.subheader("Agent 事件漏斗（协作活跃度）")
    fig_fun = px.bar(
        funnel_df,
        x="事件量",
        y="阶段",
        orientation="h",
        title="多 Agent 协作事件聚合（按阶段，条形图便于各版本 Plotly 展示）",
    )
    st.plotly_chart(fig_fun, use_container_width=True)

    st.subheader("掌握度分布")
    mastery_info = (summary or {}).get("mastery") if summary else None
    if not mastery_info or mastery_info.get("per_knowledge") is None:
        prog = st.session_state.orchestrator.get_learner_progress(st.session_state.learner_id)
        if prog.get("status") == "no_data":
            st.info("暂无掌握度数据：请先完成答题或聊天，或启动 API 后刷新。")
        else:
            model = st.session_state.orchestrator.learner_model_manager.get_or_create_model(
                st.session_state.learner_id
            )
            rows = []
            for kid, node in knowledge_graph.nodes.items():
                stt = model.get_state(kid)
                rows.append({"知识点": node.name, "掌握度": float(stt.mastery)})
            mdf = pd.DataFrame(rows).sort_values("掌握度")
            fig_m = px.histogram(
                mdf,
                x="掌握度",
                nbins=12,
                title="知识点掌握度分布（当前学习者）",
            )
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(mdf, use_container_width=True, hide_index=True)
    else:
        buckets = mastery_info.get("buckets") or {}
        if buckets:
            bdf = pd.DataFrame([{"区间": k, "知识点数": v} for k, v in buckets.items()])
            fig_b = px.bar(bdf, x="区间", y="知识点数", title="掌握度区间分布")
            st.plotly_chart(fig_b, use_container_width=True)
        pk = mastery_info.get("per_knowledge") or []
        if pk:
            pdf = pd.DataFrame(pk)
            fig_h = px.histogram(pdf, x="mastery", nbins=12, title="掌握度直方图")
            st.plotly_chart(fig_h, use_container_width=True)
            st.dataframe(pdf, use_container_width=True, hide_index=True)

    if show_raw and summary:
        st.json(summary)

# 自动跳转到进度页面（如果用户点击了快速查看）
if st.session_state.show_progress:
    js = "window.scrollTo(0, 0); document.querySelector('[data-testid=\"stTab\"]:nth-child(3)').click();"
    st.components.v1.html(f"<script>{js}</script>", height=0)
    st.session_state.show_progress = False