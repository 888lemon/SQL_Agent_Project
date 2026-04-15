import streamlit as st
import pandas as pd
import os
import json
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from app.config.settings import load_settings
from app.core.cache import ProSemanticCache
from app.core.agent import SQLProAgent
from app.db.factory import DBFactory
# 1. 基础配置
load_dotenv()
settings = load_settings()
st.set_page_config(page_title="SQL-Pro-Agent 旗舰版", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --panel: rgba(255,255,255,0.82);
        --panel-strong: rgba(255,255,255,0.92);
        --panel-border: rgba(88, 70, 49, 0.14);
        --ink: #1f1a17;
        --muted: #6c6258;
        --accent: #a24d2f;
        --accent-soft: rgba(162, 77, 47, 0.12);
        --info-soft: rgba(53, 94, 121, 0.12);
        --success-soft: rgba(48, 117, 90, 0.12);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(162, 77, 47, 0.16), transparent 32%),
            radial-gradient(circle at 85% 15%, rgba(53, 94, 121, 0.14), transparent 24%),
            linear-gradient(180deg, #fbf7f2 0%, #efe4d7 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1240px;
    }
    .hero-card, .status-card, .sql-card, .section-card, .metric-band {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 20px;
        padding: 1rem 1.15rem;
        box-shadow: 0 12px 34px rgba(66, 45, 24, 0.08);
        backdrop-filter: blur(8px);
    }
    .hero-card {
        padding: 1.35rem 1.4rem;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.94), rgba(249,242,235,0.76));
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.5;
    }
    .hero-kicker {
        display: inline-block;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.55rem;
        font-weight: 700;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0 0.4rem 0;
    }
    .status-pill {
        padding: 0.95rem 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--panel-border);
    }
    .status-pill strong {
        display: block;
        font-size: 0.82rem;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }
    .status-pill span {
        font-size: 1rem;
        color: var(--ink);
        font-weight: 600;
    }
    .soft-note {
        color: var(--muted);
        font-size: 0.9rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: var(--ink);
    }
    .section-copy {
        color: var(--muted);
        font-size: 0.92rem;
    }
    .sql-label {
        color: var(--muted);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .metric-band {
        margin-top: 1rem;
        padding: 0.9rem 1rem;
        background: var(--panel-strong);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
    }
    .metric-tile {
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,243,236,0.88));
        border: 1px solid rgba(88, 70, 49, 0.1);
    }
    .metric-tile strong {
        display: block;
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 0.3rem;
    }
    .metric-tile span {
        display: block;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
    }
    .chat-shell {
        margin-top: 1rem;
        padding: 0.95rem 1rem;
        background: rgba(255,255,255,0.55);
        border: 1px dashed rgba(88, 70, 49, 0.16);
        border-radius: 18px;
    }
    @media (max-width: 900px) {
        .status-grid { grid-template-columns: 1fr; }
        .metric-grid { grid-template-columns: 1fr 1fr; }
        .hero-title { font-size: 1.75rem; }
    }
    @media (max-width: 640px) {
        .metric-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 2. 初始化后端 (使用 st.cache_resource 确保单例)
@st.cache_resource
def init_agent():
    db = DBFactory.get_handler(settings.db_type, settings.db_path)
    # 语义缓存初始化
    cache = ProSemanticCache(
        model_name=settings.embedding_model_name,
        threshold=settings.cache_threshold,
        ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )

    # Agent 初始化 (内部会自动构建 Schema 索引)
    agent = SQLProAgent(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        db_handler=db,
        cache_engine=cache
    )
    return agent, cache


agent, cache = init_agent()

# --- Session State 状态管理 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 存储历史记录
if "last_data" not in st.session_state:
    st.session_state.last_data = None  # 存储最近一次查询的原始数据
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None  # 存储性能指标
if "last_sql" not in st.session_state:
    st.session_state.last_sql = None

# 3. 侧边栏：控制台与性能监测
with st.sidebar:
    st.title("⚙️ 系统控制台")
    st.success(f"数据库: {os.path.basename(settings.db_path)}")
    if settings.has_valid_api_key:
        st.caption("LLM API: 已配置")
    else:
        st.error("LLM API: 未配置。请在 .env 中填写真实 DEEPSEEK_API_KEY。")

    # 状态概览
    if st.session_state.last_metrics:
        m = st.session_state.last_metrics
        st.divider()
        st.subheader("⏱️ 上次查询性能")
        st.metric("总耗时", f"{m['total_latency']}s")
        st.progress(min(m['total_latency'] / 5.0, 1.0))  # 假设 5s 为慢查询阈值

        cols = st.columns(2)
        cols[0].caption("RAG 检索")
        cols[0].write(f"{m['retrieval_time']}s")
        cols[1].caption("自愈重试")
        cols[1].write(f"{m['retry_count']} 次")

    st.divider()
    if st.button("🗑️ 清空聊天记录", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_data = None
        st.session_state.last_metrics = None
        st.session_state.last_sql = None
        st.rerun()

    if st.button("🧹 强制刷新语义缓存", use_container_width=True):
        cache.clear()
        st.warning("语义索引已重建")

    # 实时日志预览 (仅展示最近 5 条)
    if os.path.exists("metrics.log"):
        with st.expander("📝 实时度量日志"):
            with open("metrics.log", "r", encoding="utf-8") as f:
                logs = f.readlines()[-5:]
                for log in reversed(logs):
                    st.json(json.loads(log))

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">SQL 智能体工作台</div>
        <div class="hero-title">工业级自愈式 SQL Agent</div>
        <div class="hero-subtitle">DeepSeek 驱动的结构化 Text-to-SQL 工作台，支持动态 Schema RAG、Join 提示、自愈修正、结果导出与离线评测。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

llm_status = "已配置" if settings.has_valid_api_key else "未配置"
cache_status = "就绪"
schema_mode = "表 + 列 + Join"
st.markdown(
    f"""
    <div class="status-card">
        <div class="status-grid">
            <div class="status-pill"><strong>LLM API</strong><span>{llm_status}</span></div>
            <div class="status-pill"><strong>Schema 模式</strong><span>{schema_mode}</span></div>
            <div class="status-pill"><strong>语义缓存</strong><span>{cache_status}</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_values = st.session_state.last_metrics or {
    "total_latency": "-",
    "retrieval_time": "-",
    "db_execution_time": "-",
    "retry_count": "-"
}
st.markdown(
    f"""
    <div class="metric-band">
        <div class="metric-grid">
            <div class="metric-tile"><strong>总耗时</strong><span>{metric_values['total_latency']}s</span></div>
            <div class="metric-tile"><strong>RAG 检索</strong><span>{metric_values['retrieval_time']}s</span></div>
            <div class="metric-tile"><strong>数据库执行</strong><span>{metric_values['db_execution_time']}s</span></div>
            <div class="metric-tile"><strong>自愈重试</strong><span>{metric_values['retry_count']} 次</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not settings.has_valid_api_key:
    st.warning("当前只完成了本地索引初始化，未配置可用的 LLM API Key，提交问题时会失败。")

# 展示对话历史
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">对话工作区</div>
        <div class="section-copy">输入业务问题后，系统会先做 Schema 检索，再生成 SQL 并返回结果摘要。你可以直接查看最终 SQL 和执行结果。</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["q"])
    with st.chat_message("assistant"):
        st.markdown(msg["a"])

# 输入框
question = st.chat_input("输入业务问题，例如：谁是业绩最好的员工？")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在检索 Schema 并生成 SQL..."):
            if not settings.has_valid_api_key:
                st.error("未检测到可用的 DEEPSEEK_API_KEY，请更新 .env 后重启应用。")
                st.stop()

            # 调用 agent 的 ask 方法
            result = agent.ask(question, history=st.session_state.chat_history)

            final_ans = result["answer"]
            raw_data = result["data"]
            generated_sql = result.get("sql")
            metrics = result.get("metrics")

            # 渲染回答
            st.markdown(final_ans)
            if agent.last_error:
                st.caption(f"最近一次 API 错误: {agent.last_error}")
            if generated_sql:
                st.markdown('<div class="sql-card"><div class="sql-label">已执行 SQL</div></div>', unsafe_allow_html=True)
                st.code(generated_sql, language="sql")

            # 更新状态
            st.session_state.chat_history.append({"q": question, "a": final_ans})
            st.session_state.last_data = raw_data
            st.session_state.last_metrics = metrics
            st.session_state.last_sql = generated_sql

            # 如果命中了缓存，给予视觉反馈
            if metrics and metrics.get("cache_hit"):
                st.toast("🚀 命中语义缓存！响应时间缩短 90%+", icon="🔥")
st.markdown('</div>', unsafe_allow_html=True)

# 5. 数据预览与导出区域
if st.session_state.last_data is not None:
    st.divider()
    left_col, right_col = st.columns([1.35, 1])
    with left_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">查询结果预览</div>
                <div class="section-copy">这里展示最近一次 SQL 执行返回的数据结果，可直接导出为 Excel 报表。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        if st.session_state.last_sql:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">最终 SQL</div>
                    <div class="section-copy">用于复核生成结果，定位错误或优化提示词。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(st.session_state.last_sql, language="sql")

    try:
        raw = st.session_state.last_data
        # 解析 DBHandler 返回的 {"columns": [], "data": []} 格式
        if isinstance(raw, dict) and "columns" in raw:
            df = pd.DataFrame(raw["data"], columns=raw["columns"])
        else:
            df = pd.DataFrame(raw)

        if df.empty:
            st.info("查询成功，但数据库中未找到符合条件的数据。")
        else:
            # 数据展示
            #st.dataframe(df, use_container_width=True, hide_index=True)
            st.dataframe(df, width='stretch', hide_index=True)
            # Excel 导出
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Result')

            st.download_button(
                label="📥 导出为 Excel 报表",
                data=output.getvalue(),
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                #use_container_width=True
                width = 'stretch'
            )

    except Exception as e:
        st.error(f"表格渲染失败: {e}")
        with st.expander("查看原始 JSON 数据"):
            st.write(st.session_state.last_data)

# 底部架构透视
st.divider()
with st.expander("🔍 系统架构透视（可观测性）"):
    tab1, tab2, tab3 = st.tabs(["当前模式（动态 Schema）", "最终 SQL", "性能指标定义"])
    with tab1:
        st.info("以下为 Agent 根据当前问题动态检索到的表、列与 Join 上下文：")
        st.code(getattr(agent, "last_schema_context", agent.schema), language="sql")
    with tab2:
        if st.session_state.last_sql:
            st.code(st.session_state.last_sql, language="sql")
        else:
            st.caption("当前还没有可展示的 SQL。")
    with tab3:
        st.markdown("""
        - **检索耗时（Retrieval Time）**：Faiss 向量库检索相关数据表的耗时。
        - **LLM 生成耗时（LLM Generation）**：DeepSeek 模型生成 SQL 与总结答案的总耗时。
        - **数据库执行耗时（DB Execution）**：SQLite 引擎执行物理查询的耗时。
        - **缓存命中（Cache Hit）**：语义相似度 > 0.95 时直接复用结果并跳过计算。
        """)
