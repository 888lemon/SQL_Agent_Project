import time
import re
import os
from pathlib import Path
from app.core.retriever import SchemaRetriever
from app.utils.logger import log_metrics
from .router import IntentRouter
from .security import SQLSecurityAudit
from .llm_client import LLMService


class QueryMetrics:
    """性能度量收集器"""

    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            "intent": "query",  # 默认为查询
            "cache_hit": False,
            "retrieval_time": 0.0,
            "llm_generation_time": 0.0,
            "db_execution_time": 0.0,
            "total_latency": 0.0,
            "retry_count": 0
        }

    def merge_cache_stats(self, cache_stats: dict):
        self.stats["cache_hit_rate"] = cache_stats.get("hit_rate", 0.0)
        self.stats["cache_size"] = cache_stats.get("size", 0)
        self.stats["cache_evictions"] = cache_stats.get("evictions", 0)
        self.stats["cache_expired"] = cache_stats.get("expired", 0)

    def stop(self):
        self.stats["total_latency"] = round(time.time() - self.start_time, 3)
        # 舍入处理方便日志阅读
        for key in ["retrieval_time", "llm_generation_time", "db_execution_time"]:
            self.stats[key] = round(self.stats[key], 3)
        return self.stats


class SQLProAgent:
    def __init__(self, api_key, base_url, db_handler, cache_engine):
        self.api_key = api_key
        self.base_url = base_url
        self.db = db_handler
        self.cache = cache_engine
        self.last_error = None
        self.last_error_type = None
        self.llm = LLMService(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=3,
            backoff_base_seconds=0.6,
        )

        # 1. 初始化意图路由
        self.router = IntentRouter(api_key, base_url)

        # 2. 初始化 Schema RAG 检索器
        self.retriever = SchemaRetriever(db_handler)
        base_dir = Path(__file__).resolve().parents[1]
        metadata_path = base_dir / "schema" / "metadata.yaml"
        few_shot_path = base_dir / "data" / "few_shot.json"

        self.retriever.build_index(str(metadata_path) if metadata_path.exists() else None)
        if few_shot_path.exists():
            self.retriever.build_example_index(str(few_shot_path))

        self.schema = self.db.get_table_info()
        self.last_schema_context = self.schema
        self.last_sql = None
        print(f"✅ 核心引擎初始化成功 (已启用 Intent-Routing, Schema-RAG 与 Observability)。")

    def _call_llm(self, prompt: str):
        """内部调用 LLM"""
        result = self.llm.call(prompt, temperature=0.0)
        if result.ok:
            self.last_error = None
            self.last_error_type = None
            return result.content

        self.last_error_type = result.error_type
        self.last_error = f"{result.error_type}: {result.error_message}"
        print(f"API调用异常: {self.last_error} (attempts={result.attempts})")
        return f"API ERROR: {self.last_error}"

    def _parse_sql(self, text: str):
        """解析 SQL 逻辑"""
        sql_match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql = sql_match.group(1)
        else:
            sql = re.sub(r".*?(SELECT)", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
        sql = " ".join(sql.split())
        return sql.strip().rstrip(';')

    def ask(self, question: str, history: list = None, max_retries=3):
        metrics = QueryMetrics()

        # --- 步骤 1: 意图识别 (Intent Routing) ---
        intent = self.router.classify(question)
        metrics.stats["intent"] = intent

        if intent == "chat":
            # 闲聊逻辑：直接回复，不查库
            ans = self._call_llm(
                f"用户向你打招呼或闲聊: {question}。请作为专业的 SQL 数据分析助手友好回应，引导用户进行数据查询。")
            final_stats = metrics.stop()
            log_metrics(question, final_stats)
            return {"answer": ans, "data": None, "metrics": final_stats}

        if intent == "unsupported":
            ans = "抱歉，我目前专注于 Northwind 数据库的数据分析（如订单、员工、产品等）。您的请求超出了我的处理范围。"
            final_stats = metrics.stop()
            log_metrics(question, final_stats)
            return {"answer": ans, "data": None, "metrics": final_stats}

        # --- 步骤 2: 语义缓存检查 ---
        cached = self.cache.query(question)
        metrics.merge_cache_stats(self.cache.get_stats())
        if cached:
            metrics.stats["cache_hit"] = True
            final_stats = metrics.stop()
            log_metrics(question, final_stats)
            self.last_sql = cached.get("sql")
            return {
                "answer": cached["answer"],
                "data": cached.get("data"),
                "sql": cached.get("sql"),
                "metrics": final_stats
            }

        # --- 步骤 3: 动态 Schema 与 Few-shot 检索 (RAG) ---
        t_rag_start = time.time()
        schema_context = self.retriever.get_relevant_schema(
            question,
            top_k_tables=3,
            top_k_columns=8,
            top_k_joins=5
        )
        few_shot_examples = self.retriever.get_few_shot_examples(question)
        metrics.stats["retrieval_time"] = time.time() - t_rag_start
        dynamic_schema = schema_context["schema_prompt"]
        self.last_schema_context = dynamic_schema

        # --- 步骤 4: 构造上下文 ---
        context_str = ""
        if history:
            for h in history[-3:]:
                context_str += f"Q: {h['q']} -> A: {h['a']}\n"

        attempt = 1
        bad_sql = ""
        last_error = ""

        # --- 步骤 5: SQL 生成与自愈循环 ---
        while attempt <= max_retries:
            metrics.stats["retry_count"] = attempt - 1

            if attempt == 1:
                prompt = f"""你是一个专业的 SQLite 专家。
{few_shot_examples}

请参考上述示例和以下结构化 Schema 上下文编写 SQL 来回答问题：
{dynamic_schema}

[对话背景]:
{context_str}

问题: {question}
[注意]: 
1. 仅输出一条完整的 SQL，不要解释。
2. 确保表名带空格时使用双引号，例如 "Order Details"。
3. 如果涉及金额计算，请记得考虑 Discount（折扣）。
4. 优先使用 [Join Hints] 中给出的关联路径，不要臆造表关系。
5. 优先参考 [Business Metrics] 中的指标定义，特别是销售额、订单量、库存相关问题。
6. 仅选择解决问题所需的列，避免 SELECT *。"""
            else:
                prompt = f"""之前的 SQL 执行报错了，请修正。
[结构化 Schema 上下文]: {dynamic_schema}
[错误 SQL]: {bad_sql}
[错误信息]: {last_error}
[修正要求]: 请检查表名是否使用了双引号、字段名是否正确、Join 是否与提示的关联路径一致。只返回修正后的 SQL。"""

            t_llm_start = time.time()
            raw_content = self._call_llm(prompt)
            metrics.stats["llm_generation_time"] += (time.time() - t_llm_start)

            if "API ERROR" in raw_content:
                error_stats = metrics.stop()
                return {"answer": "服务繁忙，请稍后再试。", "data": None, "metrics": error_stats}

            sql = self._parse_sql(raw_content)
            self.last_sql = sql
            if not SQLSecurityAudit.is_safe(sql):
                final_stats = metrics.stop()
                log_metrics(question, {**final_stats, "error": "UNSAFE_SQL_BLOCKED"})
                return {
                    "answer": "请求已被安全策略拦截。当前仅允许只读查询，请调整问题后重试。",
                    "data": None,
                    "sql": sql,
                    "metrics": final_stats
                }

            try:
                # 执行 SQL
                t_db_start = time.time()
                res = self.db.execute_query(sql, max_rows=200, timeout_seconds=5.0)
                metrics.stats["db_execution_time"] += (time.time() - t_db_start)

                # 生成最终总结
                t_sum_start = time.time()
                ans_prompt = f"问题: {question}\n查询结果数据: {res}\n请用中文简洁总结数据并回答问题，不要列出原始 SQL："
                final_ans = self._call_llm(ans_prompt)
                metrics.stats["llm_generation_time"] += (time.time() - t_sum_start)

                # 缓存与记录日志
                self.cache.update(question, {"answer": final_ans, "data": res, "sql": sql})
                metrics.merge_cache_stats(self.cache.get_stats())
                final_stats = metrics.stop()
                log_metrics(question, final_stats)

                return {"answer": final_ans, "data": res, "sql": sql, "metrics": final_stats}

            except Exception as e:
                bad_sql = sql
                last_error = str(e)
                print(f"❌ Attempt {attempt} 失败: {last_error}")
                attempt += 1
                time.sleep(0.5)

        # 最终重试失败
        fail_stats = metrics.stop()
        log_metrics(question, {**fail_stats, "error": "MAX_RETRIES_EXCEEDED"})
        return {
            "answer": "抱歉，经过多次重试，我暂时无法准确提取到该数据。",
            "data": None,
            "sql": self.last_sql,
            "metrics": fail_stats
        }
