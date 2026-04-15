import time
from collections import OrderedDict

import numpy as np
try:
    import faiss
except ModuleNotFoundError:
    faiss = None


class _NumpyIPIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype="float32")

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray):
        if vectors.size == 0:
            return
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query: np.ndarray, k: int):
        if self.ntotal == 0:
            return np.array([[0.0]], dtype="float32"), np.array([[-1]], dtype="int64")
        sims = self.vectors @ query[0]
        best_idx = int(np.argmax(sims))
        return (
            np.array([[float(sims[best_idx])]], dtype="float32"),
            np.array([[best_idx]], dtype="int64"),
        )


def _new_index(dim: int):
    if faiss is not None:
        return faiss.IndexFlatIP(dim)
    return _NumpyIPIndex(dim)


def _normalize(vectors: np.ndarray):
    if faiss is not None:
        faiss.normalize_L2(vectors)
        return
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

class ProSemanticCache:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        threshold=0.95,
        ttl_seconds: int = 1800,
        max_entries: int = 500,
        model=None,
    ):
        print("正在初始化工业级向量引擎...")
        self.model = model
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "未安装 sentence-transformers，无法初始化语义缓存。请安装 requirements.txt 依赖。"
                ) from exc
            self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        self.index = _new_index(dim)
        self.cache_data = OrderedDict()
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
        }

    def _encode(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text], show_progress_bar=False).astype('float32')
        _normalize(embedding)
        return embedding

    def _is_expired(self, created_at: float) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - created_at) > self.ttl_seconds

    def _rebuild_index(self):
        dim = self.model.get_sentence_embedding_dimension()
        self.index = _new_index(dim)
        if not self.cache_data:
            return
        vectors = np.vstack([entry["embedding"] for entry in self.cache_data.values()]).astype("float32")
        _normalize(vectors)
        self.index.add(vectors)

    def _evict_lru(self):
        if self.cache_data:
            self.cache_data.popitem(last=False)
            self.stats["evictions"] += 1

    def get_stats(self) -> dict:
        queries = self.stats["queries"]
        hit_rate = (self.stats["hits"] / queries) if queries > 0 else 0.0
        return {
            **self.stats,
            "size": len(self.cache_data),
            "hit_rate": round(hit_rate, 4),
        }

    def query(self, question: str):
        self.stats["queries"] += 1
        if self.index.ntotal == 0:
            self.stats["misses"] += 1
            return None

        expired_keys = [k for k, v in self.cache_data.items() if self._is_expired(v["created_at"])]
        if expired_keys:
            for key in expired_keys:
                self.cache_data.pop(key, None)
            self.stats["expired"] += len(expired_keys)
            self._rebuild_index()
            if self.index.ntotal == 0:
                self.stats["misses"] += 1
                return None

        embedding = self._encode(question)
        D, I = self.index.search(embedding, 1)

        if D[0][0] > self.threshold:
            keys = list(self.cache_data.keys())
            key = keys[I[0][0]]
            entry = self.cache_data.get(key)
            if entry is None:
                self.stats["misses"] += 1
                return None
            if self._is_expired(entry["created_at"]):
                self.cache_data.pop(key, None)
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                self._rebuild_index()
                return None
            self.cache_data.move_to_end(key)
            self.stats["hits"] += 1
            print(f">>> [Cache Hit] 相似度: {D[0][0]:.4f}")
            return entry["result"]
        self.stats["misses"] += 1
        return None

    def update(self, question: str, result: dict):
        embedding = self._encode(question)
        self.cache_data[question] = {
            "embedding": embedding[0],
            "result": result,
            "created_at": time.time(),
        }
        self.cache_data.move_to_end(question)

        while len(self.cache_data) > self.max_entries:
            self._evict_lru()
        self._rebuild_index()
        print(f">>> [Cache Updated] 当前缓存条数: {len(self.cache_data)}")

    def clear(self):
        dim = self.model.get_sentence_embedding_dimension()
        self.index = _new_index(dim)
        self.cache_data.clear()
        self.stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
        }
        print(">>> [Cache Cleared] 语义缓存已清空")
