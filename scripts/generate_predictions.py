import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.config.settings import load_settings, validate_required_settings
from app.core.agent import SQLProAgent
from app.core.cache import ProSemanticCache
from app.eval.evaluator import load_dataset
from app.db.factory import DBFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SQL predictions from the live agent for the eval dataset.")
    parser.add_argument("--dataset", default="app/data/eval_dataset.json", help="Path to evaluation dataset JSON.")
    parser.add_argument("--output", default="reports/predictions.json", help="Path to write predictions JSON.")
    return parser.parse_args()


def build_agent():
    load_dotenv()
    settings = load_settings()
    validate_required_settings(settings)

    db = DBFactory.get_handler(settings.db_type, settings.db_path)
    cache = ProSemanticCache(
        model_name=settings.embedding_model_name,
        threshold=settings.cache_threshold,
        ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )
    return SQLProAgent(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        db_handler=db,
        cache_engine=cache,
    )


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset)
    agent = build_agent()

    predictions = []
    for case in dataset:
        result = agent.ask(case["question"])
        predictions.append(
            {
                "id": case["id"],
                "question": case["question"],
                "sql": result.get("sql"),
                "answer": result.get("answer"),
                "metrics": result.get("metrics"),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)

    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
