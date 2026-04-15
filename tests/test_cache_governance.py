import unittest
from unittest.mock import patch

import numpy as np

from app.core.cache import ProSemanticCache


class FakeModel:
    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, show_progress_bar=False):
        _ = show_progress_bar
        result = []
        for text in texts:
            if text == "q1":
                result.append([1.0, 0.0])
            elif text == "q2":
                result.append([0.0, 1.0])
            elif text == "q3":
                result.append([0.7, 0.7])
            else:
                result.append([1.0, 0.0])
        return np.array(result, dtype="float32")


class FakeIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype="float32")

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query, k):
        sims = self.vectors @ query[0]
        best_idx = int(np.argmax(sims))
        return np.array([[float(sims[best_idx])]]), np.array([[best_idx]])


def fake_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms


class CacheGovernanceTestCase(unittest.TestCase):
    def setUp(self):
        patches = [
            patch("app.core.cache._new_index", side_effect=lambda dim: FakeIndex(dim)),
            patch("app.core.cache._normalize", side_effect=fake_normalize),
        ]
        self._patches = patches
        for p in patches:
            p.start()
        self.cache = ProSemanticCache(threshold=0.8, ttl_seconds=60, max_entries=2, model=FakeModel())

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()

    def test_lru_eviction_when_max_entries_reached(self):
        self.cache.update("q1", {"answer": "a1"})
        self.cache.update("q2", {"answer": "a2"})
        _ = self.cache.query("q1")  # q1 recent
        self.cache.update("q3", {"answer": "a3"})  # should evict q2

        self.assertIsNone(self.cache.query("q2"))
        self.assertEqual(1, self.cache.get_stats()["evictions"])

    def test_ttl_expiration_removes_entry(self):
        self.cache.update("q1", {"answer": "a1"})
        self.cache.cache_data["q1"]["created_at"] -= 120
        self.assertIsNone(self.cache.query("q1"))
        self.assertEqual(1, self.cache.get_stats()["expired"])

    def test_hit_rate_stats(self):
        self.cache.update("q1", {"answer": "a1"})
        self.assertIsNotNone(self.cache.query("q1"))
        self.assertIsNone(self.cache.query("q2"))
        stats = self.cache.get_stats()
        self.assertEqual(2, stats["queries"])
        self.assertEqual(1, stats["hits"])
        self.assertEqual(0.5, stats["hit_rate"])


if __name__ == "__main__":
    unittest.main()
