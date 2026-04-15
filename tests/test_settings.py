import os
import unittest

from app.config.settings import AppSettings, validate_required_settings


class SettingsTestCase(unittest.TestCase):
    def test_placeholder_key_is_invalid(self):
        settings = AppSettings(
            deepseek_api_key="your_api_key_here",
            deepseek_base_url="https://api.deepseek.com",
            db_type="sqlite",
            db_path="app/data/northwind.db",
            embedding_model_name="all-MiniLM-L6-v2",
            cache_threshold=0.95,
            cache_ttl_seconds=1800,
            cache_max_entries=500,
        )
        self.assertFalse(settings.has_valid_api_key)

    def test_realistic_key_is_valid(self):
        settings = AppSettings(
            deepseek_api_key="sk-demo-key",
            deepseek_base_url="https://api.deepseek.com",
            db_type="sqlite",
            db_path="app/data/northwind.db",
            embedding_model_name="all-MiniLM-L6-v2",
            cache_threshold=0.95,
            cache_ttl_seconds=1800,
            cache_max_entries=500,
        )
        self.assertTrue(settings.has_valid_api_key)

    def test_validate_required_settings_raises(self):
        settings = AppSettings(
            deepseek_api_key="",
            deepseek_base_url="https://api.deepseek.com",
            db_type="sqlite",
            db_path="app/data/northwind.db",
            embedding_model_name="all-MiniLM-L6-v2",
            cache_threshold=0.95,
            cache_ttl_seconds=1800,
            cache_max_entries=500,
        )
        with self.assertRaises(ValueError):
            validate_required_settings(settings)


if __name__ == "__main__":
    unittest.main()
