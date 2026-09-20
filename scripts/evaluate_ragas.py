"""Run RAGAS evaluation for the group RAG pipeline.

This script intentionally imports RAGAS only inside ``main`` so normal unit tests
can run in lightweight environments. Install project dependencies and configure
LLM keys in ``.env`` before running:

    python scripts/evaluate_ragas.py
"""

from __future__ import annotations

import argparse
import json
from numbers import Number
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    SAFE_REFUSAL,
    SYSTEM_PROMPT,
    call_llm,
    format_context,
    reorder_for_llm,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
DEFAULT_OUTPUT = ROOT / "group_project" / "evaluation" / "ragas_scores.json"


def _answer_from_sources(question: str, sources: list[dict]) -> str:
    if not sources:
        return SAFE_REFUSAL
    context = format_context(reorder_for_llm(sources))
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Trả lời bằng tiếng Việt. Trích dẫn bằng nhãn [Document n] tương ứng."
    )
    try:
        return call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        return SAFE_REFUSAL


def _run_pipeline_cases(cases: list[dict], *, top_k: int, use_reranking: bool) -> list[dict]:
    rows = []
    for case in cases:
        question = case["question"]
        score_threshold = -1.0 if not use_reranking else None
        if score_threshold is None:
            sources = retrieve(question, top_k=top_k, use_reranking=use_reranking)
        else:
            sources = retrieve(
                question,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking,
            )
        rows.append(
            {
                "question": question,
                "answer": _answer_from_sources(question, sources),
                "contexts": [source["content"] for source in sources],
                "ground_truth": case["expected_answer"],
                "expected_context": case["expected_context"],
                "source_document": case.get("source_document", ""),
            }
        )
    return rows


def _average_scores(frame) -> dict[str, float]:
    scores = {}
    for column in frame.columns:
        values = [
            value
            for value in frame[column].tolist()
            if isinstance(value, Number) and not isinstance(value, bool)
        ]
        if values:
            scores[column] = round(mean(values), 4)
    return scores


def evaluate_config(rows: list[dict], metrics: list) -> tuple[dict[str, float], list[dict]]:
    from datasets import Dataset
    from ragas import evaluate

    dataset = Dataset.from_list(rows)
    result = evaluate(dataset, metrics=metrics, raise_exceptions=False)
    frame = result.to_pandas()
    return _average_scores(frame), frame.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(override=True)

    try:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RAGAS is not installed. Run `python -m pip install -e .` "
            "or install the project dependencies before evaluation."
        ) from exc

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    configs = {
        "dense_only": _run_pipeline_cases(cases, top_k=args.top_k, use_reranking=False),
        "hybrid_rrf": _run_pipeline_cases(cases, top_k=args.top_k, use_reranking=True),
    }

    payload = {"top_k": args.top_k, "num_cases": len(cases), "configs": {}}
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for name, rows in configs.items():
        scores, per_row = evaluate_config(rows, metrics)
        payload["configs"][name] = {"scores": scores, "rows": per_row}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
