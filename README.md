# SQL-Pro-Agent

工业级自愈式 Text-to-SQL 智能体。  
项目支持自然语言问数、动态 Schema 检索、SQL 安全审计、自动重试修复、语义缓存与离线评测。

## 核心能力

- 动态 `Schema-RAG`：按问题检索相关表/列/Join/Few-shot，减少无关上下文。
- 意图路由：区分 `chat / query / unsupported`，降低不必要的 SQL 生成调用。
- 自愈执行：SQL 执行失败后带错误信息进行多轮修复重试。
- LLM 稳定性增强：统一 `LLMService`，支持错误分级与指数退避重试。
- 安全与护栏：AST 风险拦截 + 查询 `LIMIT` 注入 + SQLite 查询超时中断。
- 缓存治理：支持 `TTL + max_entries + LRU`，并输出命中率/淘汰/过期指标。
- 可观测性：将关键链路指标写入 `metrics.log`（JSON 行）。

## 系统流程（一次问答）

1. 用户在 `web_ui.py` 或 API 提交问题。
2. `IntentRouter` 做意图识别。
3. 查询语义缓存（命中则直接返回）。
4. 未命中时执行 Schema/Few-shot 检索。
5. LLM 生成 SQL，安全审计后执行数据库查询。
6. LLM 基于结果做中文总结。
7. 写入缓存并记录 metrics。

## 目录与文件说明

```text
SQL_Agent_Project/
├── app/
│   ├── api/
│   │   ├── endpoints.py      # FastAPI 路由: /api/query
│   │   └── schemas.py        # 统一请求模型 QueryRequest（兼容 question/text）
│   ├── config/
│   │   └── settings.py       # 环境变量加载与配置对象
│   ├── core/
│   │   ├── agent.py          # 主流程编排：路由/检索/生成/执行/总结/打点
│   │   ├── router.py         # 意图识别路由
│   │   ├── llm_client.py     # LLM 调用封装（错误分级+重试退避）
│   │   ├── retriever.py      # Schema 与 few-shot 检索
│   │   ├── cache.py          # 语义缓存（TTL/LRU/容量）
│   │   └── security.py       # SQL 安全审计与 LIMIT 护栏
│   ├── db/
│   │   ├── base.py           # DB 抽象接口
│   │   ├── handler.py        # SQLite/MySQL 处理器实现
│   │   └── factory.py        # DB handler 工厂
│   ├── eval/
│   │   ├── evaluator.py      # 检索/结构/执行评测逻辑
│   │   └── __init__.py
│   ├── schema/
│   │   └── metadata.yaml     # 业务语义元数据（表/列/Join/指标）
│   ├── data/
│   │   ├── few_shot.json     # Few-shot 示例
│   │   └── eval_dataset.json # 离线评测集
│   └── utils/
│       └── logger.py         # metrics 日志
├── scripts/
│   ├── check_env.py          # 环境依赖自检
│   ├── generate_predictions.py # 生成预测 SQL/答案
│   └── run_eval.py           # 离线评测入口
├── tests/
│   ├── test_api_schemas.py
│   ├── test_cache_governance.py
│   ├── test_evaluator.py
│   ├── test_metadata_yaml.py
│   ├── test_schema_graph.py
│   ├── test_security.py
│   └── test_settings.py
├── app/main.py               # FastAPI 应用入口（/health, /ask）
├── web_ui.py                 # Streamlit UI 入口
├── requirements.txt
├── requirements-dev.txt
└── Dockerfile
```

## 配置项（`.env`）

至少需要以下配置：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DB_TYPE=sqlite
DB_PATH=app/data/northwind.db
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CACHE_THRESHOLD=0.95
CACHE_TTL_SECONDS=1800
CACHE_MAX_ENTRIES=500
```

## 运行方式

### 1) 本地运行（pip/venv）

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run web_ui.py
```

### 2) Conda 运行（Windows 推荐）

如果你用 Conda 管环境，建议底层包优先 conda 安装，减少二进制冲突：

```bash
conda create -n sql-agent python=3.11 -y
conda activate sql-agent
conda install -y numpy=1.26 pandas=2.2
pip install -r requirements.txt
streamlit run web_ui.py
```

### 3) API 服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

接口说明：

- `GET /health`: 健康检查
- `POST /ask`: 请求体兼容 `{ "question": "..." }` 或 `{ "text": "..." }`
- `POST /api/query`: 同样兼容 `question/text`

## 评测与脚本

```bash
# 环境检查
python scripts/check_env.py

# 生成预测
python scripts/generate_predictions.py --output reports/predictions.json

# 运行评测
python scripts/run_eval.py --predictions reports/predictions.json --output reports/eval_report.json
```

## 测试

```bash
python -m unittest discover -s tests -p "test_*.py"
```

如本地缺失 `faiss` / `sentence_transformers`，可先跑不依赖它们的最小集合：

```bash
python -m unittest tests.test_api_schemas tests.test_llm_client tests.test_settings
```

## 指标日志示例（`metrics.log`）

```json
{
  "timestamp": "2026-04-13T20:30:00",
  "question": "谁是销售额最高的员工？",
  "intent": "query",
  "cache_hit": false,
  "cache_hit_rate": 0.33,
  "cache_size": 42,
  "cache_evictions": 3,
  "cache_expired": 5,
  "retrieval_time": 0.41,
  "llm_generation_time": 1.92,
  "db_execution_time": 0.07,
  "retry_count": 1,
  "total_latency": 2.48
}
```
