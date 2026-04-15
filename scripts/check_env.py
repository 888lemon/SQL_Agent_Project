import importlib
import platform
import sys


REQUIRED_PACKAGES = {
    "numpy": "Core numeric runtime",
    "pandas": "Query execution and dataframe export",
    "pyarrow": "Pandas binary compatibility",
    "sqlglot": "SQL parsing and security audit",
    "faiss": "Vector retrieval index",
    "sentence_transformers": "Embedding model runtime",
    "streamlit": "Web UI",
    "fastapi": "API runtime",
}


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print()

    failures = []
    for module_name, description in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {module_name:<22} {version:<12} {description}")
        except Exception as exc:
            failures.append((module_name, str(exc)))
            print(f"[FAIL] {module_name:<20} {description}")
            print(f"       {exc}")

    print()
    if failures:
        print("Environment check failed.")
        print("Suggested fix:")
        print("1. Create a fresh virtual environment.")
        print("2. Install with: pip install -r requirements-dev.txt")
        print("3. Re-run: python scripts/check_env.py")
        raise SystemExit(1)

    print("Environment check passed.")


if __name__ == "__main__":
    main()
