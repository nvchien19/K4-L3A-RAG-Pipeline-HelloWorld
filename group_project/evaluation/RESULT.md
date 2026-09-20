# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | RAGAS target: `ragas==0.4.3`; current checkout did not have `ragas` installed |
| Evaluator model                    | Configure via RAGAS/OpenAI-compatible evaluator before running `scripts/evaluate_ragas.py` |
| Generator model                    | `LLM_PROVIDER`/`LLM_MODEL` from `.env`; Task 10 supports OpenAI, Gemini, Anthropic |
| Embedding model                    | `.env.example`: `EMBEDDING_PROVIDER=sentence_transformers`, `EMBEDDING_MODEL=BAAI/bge-m3` |
| Corpus version/commit              | `da0e209`, đã làm sạch boilerplate bằng `scripts/clean_standardized.py` (225.034 → 189.072 ký tự, giảm 16.0%); 471 chunks sau khi index lại |
| Golden dataset size                | 18 grounded cases |
| `top_k`                            | 5 |
| Fallback threshold and calibration | Recommended `SCORE_THRESHOLD=0.447` (hiệu chỉnh lại sau khi làm sạch data và index lại); tách hoàn toàn in-domain/out-of-domain, 0 false fallback và 0 missed fallback. Giá trị cũ `0.424` đo trên corpus còn boilerplate nên đã hết hiệu lực. |

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

- Cấu hình tốt hơn: **chưa kết luận được.** RAGAS chưa chạy. Ở phép đo retrieval-only đã chạy thật (recall@5 theo `source_document`, 18 case), **A và B hoà 17/18 (94.4%)** và cùng miss `legal-09` — xem `reports/retrieval_ab_findings.json`.
- Evidence: RRF fusion đã được xác minh là thật sau khi sửa bug `CORPUS` của Task 6 — score `0.03126 = 1/61 + 1/63` chứng tỏ có chunk xuất hiện ở cả hai ranked list (trước khi sửa, score trần luôn đúng bằng `1/61`). Ngưỡng fallback lấy từ `reports/threshold_calibration.json` (`0.424`).
- Lưu ý: recall@5 theo `source_document` là metric thô (chỉ hỏi "có lấy đúng tài liệu không"), **không** thay thế context recall/precision của RAGAS. Hai config có thể hoà ở metric này nhưng khác nhau ở thứ hạng và độ sạch của context.
- Trade-off về latency/cost: Config A is cheaper and faster because it performs dense retrieval only. Config B adds BM25 + RRF and may call fallback, so it costs more but should improve recall on lexical/listing questions such as legal provisions and exact tourism statistics.

## Worst performers

These are cases to inspect first after running RAGAS, based on dataset difficulty notes and retrieval risk.

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Có những loại cơ sở lưu trú du lịch nào? | A/B | n/a | n/a | **miss** | n/a | retrieval (data) | **Measured, both configs miss.** Ground truth is `luat-du-lich-2017.md::chunk-140` (Điều 48). It states "cơ sở lưu trú du lịch" once as a heading then lists bare items, while Nghị định chunks repeat the phrase many times → term-frequency trap. Compounded by `"Đang theo dõi"` boilerplate (see Recommendations #1). BM25 did **not** rescue this case. |
|   2 | Điều kiện kinh doanh dịch vụ lữ hành nội địa khác gì so với lữ hành quốc tế về bằng cấp của người phụ trách? | A/B | n/a | n/a | n/a | n/a | retrieval/generation | Comparison question needs two clauses from the same legal article and careful synthesis. |
|   3 | Trong 8 tháng năm 2026, khách quốc tế đến Việt Nam bằng đường hàng không chiếm tỷ lệ bao nhiêu? | A/B | n/a | n/a | n/a | n/a | retrieval/generation | Numeric news answer requires exact percentage and adjacent transport-mode figures. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Run `scripts/evaluate_ragas.py` after installing dependencies and setting evaluator/generator API keys. | RAGAS package is not present in this checkout; no metric scores were produced. | Produces the required faithfulness, answer relevance, context recall and context precision scores. | `group_project/evaluation/ragas_scores.json` exists and contains both `dense_only` and `hybrid_rrf`. |
|        2 | ~~Xoá boilerplate khỏi `data/standardized/`~~ — **ĐÃ LÀM**, xem `scripts/clean_standardized.py`. | Web chrome scrape từ LuatVietnam: `"Đang theo dõi"` 857 lần, cộng menu điều hướng và tường đăng nhập. Tổng cộng 16.0% ký tự là rác. | Corpus giảm 225.034 → 189.072 ký tự; số chunk giảm 563 → 471. Đã verify **144/144 điều luật còn nguyên** và **18/18 expected_context của golden dataset còn ≥95% từ khoá**. | Đã re-index và re-calibrate: `SCORE_THRESHOLD` 0.424 → 0.447. Chạy `python scripts/clean_standardized.py --dry-run` để xác nhận không còn gì để cắt. |
|        4 | Xử lý riêng case liệt kê như `legal-09` (Điều 48): cân nhắc chunk theo ranh giới "Điều" để giữ nguyên đầu mục + danh sách trong cùng một chunk. | Ground truth nêu cụm từ khoá đúng **một lần** rồi liệt kê mục trần, nên thua các chunk lặp cụm từ đó nhiều lần ở cả dense lẫn BM25. | Khắc phục đúng loại câu hỏi "có những loại nào". | `legal-09` lọt top-5 ở config B. |
|        3 | Keep `SCORE_THRESHOLD=0.424` as the starting fallback threshold and recalibrate after corpus/index changes. | Calibration separates in-domain mean `0.6045` from out-of-domain mean `0.2352`. | Reduces unnecessary fallback while catching out-of-domain queries. | Rerun `scripts/calibrate_threshold.py` and compare false fallback/missed fallback counts. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| UI citation/source highlighting | Config B base UI | n/a before RAGAS run | Minimal UI-only cost | Implemented in `app.py`; sources show content, method and score for each answer. |
