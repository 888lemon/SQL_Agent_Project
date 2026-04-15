import faiss
import numpy as np
import yaml
import os
from collections import deque, defaultdict
from sentence_transformers import SentenceTransformer
import json

class SchemaRetriever:
    def __init__(self, db_handler, model_name='all-MiniLM-L6-v2'):
        self.db = db_handler
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.table_index = None
        self.column_index = None
        self.join_index = None
        self.metric_index = None
        self.table_docs = []
        self.column_docs = []
        self.join_docs = []
        self.schema_graph = {"tables": {}, "joins": []}
        self.table_names = []
        self.join_graph = defaultdict(list)
        self.metric_docs = []

    def _tokenize(self, text):
        lowered = text.lower()
        cleaned = []
        current = []
        for ch in lowered:
            if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
                current.append(ch)
            else:
                if current:
                    cleaned.append("".join(current))
                    current = []
        if current:
            cleaned.append("".join(current))
        return cleaned

    def _keyword_bonus(self, question_tokens, keywords):
        if not keywords:
            return 0.0
        normalized_keywords = {str(keyword).lower() for keyword in keywords if keyword}
        overlap = sum(1 for token in question_tokens if token in normalized_keywords)
        return overlap * 0.08

    def _encode(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        return embeddings

    def _build_index_from_docs(self, docs):
        if not docs:
            return None
        embeddings = self._encode([doc["text"] for doc in docs])
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        return index

    def _normalize_metadata_entry(self, metadata, table_name):
        entry = metadata.get(table_name, {})
        if isinstance(entry, str):
            return {"description": entry, "columns": {}, "aliases": []}
        if isinstance(entry, dict):
            return {
                "description": entry.get("description", "该表包含业务数据字段。"),
                "columns": entry.get("columns", {}),
                "aliases": entry.get("aliases", [])
            }
        return {"description": "该表包含业务数据字段。", "columns": {}, "aliases": []}

    def _prepare_metadata(self, metadata_path=None):
        """整合数据库 DDL、字段和 Join 关系，构建分层检索文档"""
        self.schema_graph = self.db.get_schema_graph()

        metadata = {}
        metric_entries = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f) or {}
        metric_entries = metadata.pop("metrics", {}) if isinstance(metadata, dict) else {}

        table_docs = []
        column_docs = []
        join_docs = []
        metric_docs = []
        self.table_names = []
        self.join_graph = defaultdict(list)

        for table_name, info in self.schema_graph["tables"].items():
            meta_entry = self._normalize_metadata_entry(metadata, table_name)
            column_names = [column["name"] for column in info["columns"]]
            aliases = ", ".join(meta_entry["aliases"])
            combined_text = (
                f"Table: {table_name}. "
                f"Description: {meta_entry['description']}. "
                f"Aliases: {aliases}. "
                f"Columns: {', '.join(column_names)}. "
                f"Schema: {info['ddl']}"
            )
            table_docs.append({
                "table": table_name,
                "text": combined_text,
                "keywords": [table_name, *meta_entry["aliases"], *column_names]
            })
            self.table_names.append(table_name)

            for column in info["columns"]:
                column_desc = meta_entry["columns"].get(column["name"], "业务字段")
                column_docs.append({
                    "table": table_name,
                    "column": column["name"],
                    "text": (
                        f"Table: {table_name}. Column: {column['name']}. "
                        f"Type: {column['type']}. Description: {column_desc}. "
                        f"PrimaryKey: {column['pk']}"
                    ),
                    "keywords": [table_name, column["name"], column_desc, *meta_entry["aliases"]]
                })

        for join in self.schema_graph["joins"]:
            join_text = (
                f"Join path: {join['left_table']}.{join['left_column']} = "
                f"{join['right_table']}.{join['right_column']}"
            )
            join_docs.append({
                "left_table": join["left_table"],
                "right_table": join["right_table"],
                "text": join_text,
                "keywords": [
                    join["left_table"],
                    join["left_column"],
                    join["right_table"],
                    join["right_column"]
                ]
            })
            self.join_graph[join["left_table"]].append((join["right_table"], join_text))
            self.join_graph[join["right_table"]].append((join["left_table"], join_text))

        for metric_name, metric_info in metric_entries.items():
            aliases = metric_info.get("aliases", [])
            formula = metric_info.get("formula", "")
            tables = metric_info.get("tables", [])
            metric_docs.append({
                "metric": metric_name,
                "text": (
                    f"Metric: {metric_name}. Aliases: {', '.join(aliases)}. "
                    f"Formula: {formula}. Related tables: {', '.join(tables)}"
                ),
                "keywords": [metric_name, *aliases, *tables],
                "tables": tables,
                "formula": formula
            })

        return table_docs, column_docs, join_docs, metric_docs

    def build_index(self, metadata_path=None):
        """构建表、字段、Join 三层向量索引"""
        self.table_docs, self.column_docs, self.join_docs, self.metric_docs = self._prepare_metadata(metadata_path)
        if not self.table_docs:
            print("⚠️ 数据库中未找到任何表，跳过索引构建。")
            return

        self.table_index = self._build_index_from_docs(self.table_docs)
        self.column_index = self._build_index_from_docs(self.column_docs)
        self.join_index = self._build_index_from_docs(self.join_docs)
        self.metric_index = self._build_index_from_docs(self.metric_docs)
        print(
            f"✅ Schema 分层索引构建完成，载入 {len(self.table_docs)} 张表、"
            f"{len(self.column_docs)} 个字段、{len(self.join_docs)} 条 Join、"
            f"{len(self.metric_docs)} 个业务指标。"
        )

    def _search_docs(self, index, docs, question, top_k):
        if index is None or not docs:
            return []
        actual_k = min(top_k, len(docs))
        q_emb = self._encode([question])
        scores, indices = index.search(q_emb, actual_k)
        question_tokens = self._tokenize(question)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = docs[idx]
            rerank_score = float(score) + self._keyword_bonus(question_tokens, doc.get("keywords", []))
            hits.append((rerank_score, doc))
        hits.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in hits]

    def _expand_join_paths(self, selected_tables):
        if len(selected_tables) < 2:
            return []

        selected_set = set(selected_tables)
        join_hints = []
        seen = set()

        for source in selected_tables:
            queue = deque([(source, [])])
            visited = {source}
            while queue:
                current, path = queue.popleft()
                if current in selected_set and current != source and path:
                    for join_text in path:
                        if join_text not in seen:
                            seen.add(join_text)
                            join_hints.append(join_text)
                    break
                for neighbor, join_text in self.join_graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [join_text]))
        return join_hints

    def get_relevant_schema(self, question, top_k_tables=3, top_k_columns=8, top_k_joins=5):
        """根据问题检索相关表、字段和 Join 关系，返回结构化上下文"""
        if self.table_index is None or not self.table_names:
            return {
                "schema_prompt": self.db.get_table_info(),
                "selected_tables": [],
                "selected_columns": [],
                "selected_joins": []
            }

        table_hits = self._search_docs(self.table_index, self.table_docs, question, top_k_tables)
        column_hits = self._search_docs(self.column_index, self.column_docs, question, top_k_columns)
        join_hits = self._search_docs(self.join_index, self.join_docs, question, top_k_joins)
        metric_hits = self._search_docs(self.metric_index, self.metric_docs, question, top_k=3)

        selected_tables = []
        seen_tables = set()
        for hit in table_hits + column_hits:
            table_name = hit["table"]
            if table_name not in seen_tables:
                seen_tables.add(table_name)
                selected_tables.append(table_name)

        for metric in metric_hits:
            for table_name in metric.get("tables", []):
                if table_name not in seen_tables and table_name in self.schema_graph["tables"]:
                    seen_tables.add(table_name)
                    selected_tables.append(table_name)

        selected_columns = []
        seen_columns = set()
        for hit in column_hits:
            key = (hit["table"], hit["column"])
            if key not in seen_columns:
                seen_columns.add(key)
                selected_columns.append(key)

        selected_join_texts = []
        seen_joins = set()
        for hit in join_hits:
            if hit["text"] not in seen_joins:
                seen_joins.add(hit["text"])
                selected_join_texts.append(hit["text"])

        for join_text in self._expand_join_paths(selected_tables):
            if join_text not in seen_joins:
                seen_joins.add(join_text)
                selected_join_texts.append(join_text)

        ddl = self.db.get_table_info(table_names=selected_tables)
        column_lines = [
            f"- {table}.{column}"
            for table, column in selected_columns
            if table in seen_tables
        ]
        join_lines = [f"- {join_text}" for join_text in selected_join_texts]
        metric_lines = [
            f"- {metric['metric']}: {metric['formula']}"
            for metric in metric_hits
        ]

        schema_prompt_parts = [
            "[Relevant Tables]",
            ", ".join(selected_tables) if selected_tables else "(none)",
            "",
            "[Relevant DDL]",
            ddl,
            "",
            "[Relevant Columns]",
            "\n".join(column_lines) if column_lines else "(none)",
            "",
            "[Business Metrics]",
            "\n".join(metric_lines) if metric_lines else "(none)",
            "",
            "[Join Hints]",
            "\n".join(join_lines) if join_lines else "(none)"
        ]
        print(f"🎯 RAG 命中相关表: {selected_tables}")
        if selected_columns:
            print(f"🧩 RAG 命中相关字段: {selected_columns}")
        if selected_join_texts:
            print(f"🔗 RAG 命中 Join: {selected_join_texts}")
        return {
            "schema_prompt": "\n".join(schema_prompt_parts),
            "selected_tables": selected_tables,
            "selected_columns": selected_columns,
            "selected_joins": selected_join_texts,
            "selected_metrics": [metric["metric"] for metric in metric_hits]
        }

    def build_example_index(self, few_shot_path):
        """为示例库构建向量索引"""
        if not os.path.exists(few_shot_path):
            self.examples = []
            return

        with open(few_shot_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)

        if not self.examples: return

        # 向量化所有示例中的问题
        example_qs = [ex["question"] for ex in self.examples]
        ex_embeddings = self._encode(example_qs)

        self.ex_index = faiss.IndexFlatIP(ex_embeddings.shape[1])
        self.ex_index.add(ex_embeddings.astype('float32'))
        print(f"✅ 成功加载 {len(self.examples)} 个 Few-Shot 示例")

    def get_few_shot_examples(self, question, top_k=2):
        """检索与当前问题最相似的 SQL 示例"""
        if not hasattr(self, 'ex_index') or not self.examples:
            return ""

        q_emb = self._encode([question])
        _, indices = self.ex_index.search(q_emb.astype('float32'), min(top_k, len(self.examples)))

        selected = [self.examples[i] for i in indices[0]]
        prompt_segment = "\n参考以下相似问题的 SQL 写法：\n"
        for ex in selected:
            prompt_segment += f"问题: {ex['question']}\nSQL: ```sql\n{ex['sql']}\n```\n"
        return prompt_segment
