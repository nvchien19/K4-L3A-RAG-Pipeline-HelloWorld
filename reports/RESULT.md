# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | RAGAS target: `ragas==0.4.3`; current checkout did not have `ragas` installed |
| Evaluator model                    | Configure via RAGAS/OpenAI-compatible evaluator before running `scripts/evaluate_ragas.py` |
| Generator model                    | `LLM_PROVIDER`/`LLM_MODEL` from `.env`; Task 10 supports OpenAI, Gemini, Anthropic |
| Embedding model                    | `.env.example`: `EMBEDDING_PROVIDER=sentence_transformers`, `EMBEDDING_MODEL=BAAI/bge-m3` |
| Corpus version/commit              | `da0e209` |
| Golden dataset size                | 18 grounded cases |
| `top_k`                            | 5 |
| Fallback threshold and calibration | Recommended `SCORE_THRESHOLD=0.424`; in-domain mean dense score `0.6045`, out-of-domain mean `0.2352`, no false fallback or missed fallback in calibration sample |

## Configurations

- **Config A - dense-only:** run `retrieve(question, top_k=5, score_threshold=-1.0, use_reranking=False)` to disable fallback and reranking, then use the same Task 10 prompt/generator.
- **Config B - hybrid + RRF:** run `retrieve(question, top_k=5, use_reranking=True)`, combining dense + BM25 with RRF and PageIndex fallback when dense confidence is below threshold.

Both configurations must use the same golden dataset, generator, evaluator, prompt and `top_k`; only retrieval strategy changes. The reproducible script is `scripts/evaluate_ragas.py`.

## Overall scores

RAGAS was not executed in this checkout because the package was absent (`ModuleNotFoundError: No module named 'ragas'`). The table is intentionally marked as not-run rather than filled with fabricated scores.

| Metric            | Config A | Config B | Delta B-A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |  not-run |  not-run |   not-run |
| Answer relevance  |  not-run |  not-run |   not-run |
| Context recall    |  not-run |  not-run |   not-run |
| Context precision |  not-run |  not-run |   not-run |
| **Average**       |  not-run |  not-run |   not-run |

Run command:

```bash
python -m pip install -e .
python scripts/evaluate_ragas.py --top-k 5
```

## A/B comparison

- Cấu hình tốt hơn: pending actual RAGAS run; expected candidate is Config B because it combines semantic and lexical evidence and keeps PageIndex fallback for low-confidence dense retrieval.
- Evidence: available pre-run evidence is the threshold calibration file `reports/threshold_calibration.json`, which separates in-domain and out-of-domain dense scores with recommended threshold `0.424`; final evidence should come from `group_project/evaluation/ragas_scores.json`.
- Trade-off về latency/cost: Config A is cheaper and faster because it performs dense retrieval only. Config B adds BM25 + RRF and may call fallback, so it costs more but should improve recall on lexical/listing questions such as legal provisions and exact tourism statistics.

## Worst performers

These are cases to inspect first after running RAGAS, based on dataset difficulty notes and retrieval risk.

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Có những loại cơ sở lưu trú du lịch nào? | A/B | n/a | n/a | n/a | n/a | retrieval | Known hard listing case; dense-only may retrieve nearby lodging-condition chunks instead of Điều 48 list. |
|   2 | Điều kiện kinh doanh dịch vụ lữ hành nội địa khác gì so với lữ hành quốc tế về bằng cấp của người phụ trách? | A/B | n/a | n/a | n/a | n/a | retrieval/generation | Comparison question needs two clauses from the same legal article and careful synthesis. |
|   3 | Trong 8 tháng năm 2026, khách quốc tế đến Việt Nam bằng đường hàng không chiếm tỷ lệ bao nhiêu? | A/B | n/a | n/a | n/a | n/a | retrieval/generation | Numeric news answer requires exact percentage and adjacent transport-mode figures. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Run `scripts/evaluate_ragas.py` after installing dependencies and setting evaluator/generator API keys. | RAGAS package is not present in this checkout; no metric scores were produced. | Produces the required faithfulness, answer relevance, context recall and context precision scores. | `group_project/evaluation/ragas_scores.json` exists and contains both `dense_only` and `hybrid_rrf`. |
|        2 | Review low-recall cases and adjust chunk size or retrieval fusion if Config B misses exact legal articles. | Golden dataset includes known hard listing/comparison cases. | Improves context recall and citation grounding. | Rerun RAGAS and compare context recall/precision deltas. |
|        3 | Keep `SCORE_THRESHOLD=0.424` as the starting fallback threshold and recalibrate after corpus/index changes. | Calibration separates in-domain mean `0.6045` from out-of-domain mean `0.2352`. | Reduces unnecessary fallback while catching out-of-domain queries. | Rerun `scripts/calibrate_threshold.py` and compare false fallback/missed fallback counts. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| UI citation/source highlighting | Config B base UI | n/a before RAGAS run | Minimal UI-only cost | Implemented in `app.py`; sources show content, method and score for each answer. |
