import unittest

import yaml


class MetadataYamlTestCase(unittest.TestCase):
    def test_metrics_and_table_aliases_exist(self):
        with open("app/schema/metadata.yaml", "r", encoding="utf-8") as file:
            metadata = yaml.safe_load(file)

        self.assertIn("metrics", metadata)
        self.assertIn("sales_amount", metadata["metrics"])
        self.assertIn("Orders", metadata)
        self.assertTrue(metadata["Orders"]["aliases"])
        self.assertIn("OrderDate", metadata["Orders"]["columns"])


if __name__ == "__main__":
    unittest.main()
