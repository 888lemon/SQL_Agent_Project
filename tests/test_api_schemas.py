import unittest

from app.api.schemas import QueryRequest


class QueryRequestTestCase(unittest.TestCase):
    def test_prefers_question_field(self):
        req = QueryRequest(question="  查询销量最高产品  ", text="旧字段")
        self.assertEqual(req.resolved_question(), "查询销量最高产品")

    def test_falls_back_to_legacy_text_field(self):
        req = QueryRequest(text="  查询最近订单  ")
        self.assertEqual(req.resolved_question(), "查询最近订单")

    def test_raises_for_empty_payload(self):
        req = QueryRequest()
        with self.assertRaises(ValueError):
            req.resolved_question()


if __name__ == "__main__":
    unittest.main()
