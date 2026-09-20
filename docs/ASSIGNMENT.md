# Phân việc nhóm — RAG Pipeline (Day 8)

Repo có 10 task được implement sẵn khung (stub `NotImplementedError`) trong `src/`. Mỗi người nhận **2 task chính** và **đóng góp chung** vào evaluation. Toàn bộ input/output phải tuân [Module contracts](MODULE_CONTRACTS.md); rubric chấm điểm ở [GRADING_RUBRIC.md](GRADING_RUBRIC.md).

## Thành viên

| Thành viên | Mã HV | Role | Branch đề xuất | Task chính |
|---|---|---|---|---|
| Vũ Văn Hà | 2A202602788 | Data lead | `task-data` | 1, 2, 3 |
| Nguyễn Văn Chiến | 2A202602926 | Indexing lead | `task-index` | 4, 5 |
| Nguyễn Hồ Nam | 2A202602589 | Fusion lead | `task-fusion` | 6, 7 |
| Nguyễn Cảnh Duy | 2A202602815 | Retrieval lead | `task-pipeline` | 8, 9 |
| Nguyễn Trọng Huy | 2A202602379 | Gen/UI lead | `task-generation` | 10 + `app.py` |

## Phân công chi tiết

### Vũ Văn Hà — Data (Tasks 1, 2, 3)

- Chọn đề tài nhóm (tham khảo [SUGGESTED_TOPICS](SUGGESTED_TOPICS.md)) và thống nhất với cả team.
- `task1_collect_legal_docs.py`: tải ≥ 3 PDF/DOCX vào `data/landing/legal/` (tên không dấu, > 1 KB).
- `task2_crawl_news.py`: crawl ≥ 5 bài vào `data/landing/news/` — mỗi JSON đủ `url`, `title`, `date_crawled`, `content_markdown`.
- `task3_convert_markdown.py`: chuẩn hoá sang `data/standardized/{legal,news}` (giữ header metadata; không tạo file trùng khi chạy lại).

**Chấp nhận (Acceptance):**
```bash
pytest tests/test_acceptance.py -q
```
riêng: `test_corpus_has_required_legal_documents`, `test_corpus_has_required_news_with_metadata`, `test_standardized_output_covers_both_source_types`.
**Ghi chú:** khối data chặn mọi pipeline; là milestone đầu tiên, ưu tiên xong trước.

### Văn Chiến — Indexing + Dense (Tasks 4, 5)

- `task4_chunking_indexing.py`: `load_documents`, `chunk_documents`, `embed_texts`, `embed_chunks`, `get_collection`, `index_to_vectorstore`.
  - Embed theo `EMBEDDING_PROVIDER` trong `.env`; ID chunk ổn định (`<doc-id>::chunk-<i>`).
  - ChromaDB: `PersistentClient` + collection cosine.
- `task5_semantic_search.py`: `semantic_search` **dùng chung** `embed_texts`/`get_collection` của Task 4; đổi cosine distance → similarity.

**Chấp nhận:**
```bash
python -m src.task4_chunking_indexing
python -m src.task5_semantic_search
pytest tests/test_contracts.py -k "chunk_documents or semantic_search" -q
```

### Nguyễn Hồ Nam — Lexical + Fusion (Tasks 6, 7)

- `task6_lexical_search.py`: `build_bm25_index` + `lexical_search` trên cùng corpus chunks VỚI Task 5.
- `task7_reranking.py`: `rerank_rrf` — công thức `sum(1 / (k + rank))`, rank bắt đầu 1, mặc định `k=60`, chỉ fuse một lần.

**Chấp nhận:**
```bash
pytest tests/test_contracts.py -k "lexical or rrf" -q
```

### Cảnh Duy — Fallback + Pipeline (Tasks 8, 9)

- `task8_pageindex_vectorless.py`: đọc `PAGEINDEX_API_KEY`, upload + cache IDs, `pageindex_search` trả `SearchResult` method `pageindex`; xử lý lỗi/timeout.
- `task9_retrieval_pipeline.py`: `retrieve` — dense + BM25 → RRF đúng một lần; so sánh **dense cosine gốc** với `score_threshold` mới quyết định fallback; fallback lỗi trả hybrid, không crash.
- Hiệu chỉnh `SCORE_THRESHOLD` bằng query in-domain và out-of-domain.

**Chấp nhận:**
```bash
pytest tests/test_contracts.py -k "retrieve" -q
```

### Trọng Huy — Generation + UI + Integration (Task 10, `app.py`)

- `task10_generation.py`: `reorder_for_llm` (không mutate), `format_context` (có title + source), `call_llm` dispatch theo `LLM_PROVIDER`, `generate_with_citation`; safe refusal khi thiếu evidence.
- `app.py`: thay placeholder bằng `generate_with_citation(query, top_k)`; hiển thị answer, `sources`, `retrieval_method`, score.
- Chịu trách nhiệm **end-to-end**: đảm bảo `RetrievalSource` map đúng (`hybrid`/`pageindex`/`none`).

**Chấp nhận:**
```bash
pytest tests/test_contracts.py -k "reorder or generation" -q
streamlit run app.py
```

## Evaluation — việc chung ai chủ trì

| Hạng mục | Chủ trì | Mô tả |
|---|---|---|
| Golden dataset ≥ 15 Q&A | Vũ Văn Hà góp phần legal, cả team góp, Trọng Huy tổng hợp | Ghi vào `group_project/evaluation/golden_dataset.json` |
| Script đánh giá 4 metric (ragas) | Trọng Huy | faithfulness, answer relevance, context recall, context precision |
| Config A dense-only vs B hybrid+RRF | Văn Chiến + Cảnh Duy chạy, Nam/Trọng Huy phân tích | Cùng dataset/generator/prompt/top_k, chỉ khác retrieval |
| Hiệu chỉnh threshold & ghi lại | Cảnh Duy | Ghi vào `RESULT.md` phần calibration |
| Hoàn thiện `group_project/evaluation/RESULT.md` | Trọng Huy | Không còn `TODO`, đủ 4 heading acceptance test yêu cầu |
| Báo cáo cá nhân | Từng người | Copy `reports/INDIVIDUAL_REPORT.md` → `reports/<ma-hv>-<ten>.md` |

## Thứ tự làm việc (dependency)

```
[T1 legal]────┐
[T2 news]─────┴→ [T3 standardize] → [T4 chunk/embed/index] → [T5 dense]──┐
                                                          [T6 bm25]──→ [T7 RRF]──→ [T9 retrieve] → [T10 generation] → [app.py]
            [T8 pageindex] ──────────────────────────────────────────────────────→↑(fallback)                        → [evaluation]
```

- Vũ Văn Hà làm trước (milestone 0–1). Văn Chiến/Nguyễn Hồ Nam nhận ra khung Task 4–7 sớm (Văn Chiến cần Vũ Văn Hà xong data).
- Cảnh Duy làm Task 8 song song với Nguyễn Hồ Nam; Task 9 chờ 5–7 trả đủ.
- Trọng Huy viết `reorder_for_llm`/`format_context` sớm (không phụ thuộc data), `call_llm`/`generate_with_citation` chờ Task 9.
- Evaluation chạy cuối khi pipeline end-to-end xong.

## Thời lượng tham chiếu (lộ trình 3 giờ)

| Mốc | Thời gian | Người nộp |
|---|---|---|
| Setup + chọn đề tài | 0–10' | Cả team (Vũ Văn Hà chủ trì tên đề tài) |
| Data hoàn tất | 10–35' | Vũ Văn Hà |
| Index + search chạy | 35–65' | Văn Chiến (Nam chuẩn bị BM25) |
| RRF + fallback | 65–90' | Nam + Cảnh Duy |
| Generation + UI | 90–120' | Trọng Huy |
| Evaluation + reports | 120–150' | Cả team, Trọng Huy tổng hợp |
| Test, demo, push | 150–180' | Cả team |

## Quy tắc chung

- `id` duy nhất và ổn định; siêu dữ liệu (`source`, `title`, `doc_type`, `url`, `chunk_index`) giữ xuyên pipeline.
- Task 4 và 5 dùng chung embedding model + `embed_texts()`.
- RRF chỉ fuse một lần; fallback chỉ so sánh với dense cosine gốc.
- Không commit `.env`, API key, cache. `.env` ở bản local, nộp `.env.example`.
- Cuối buổi chạy đủ: `pytest -q`, viết individual report, kiểm tra repo không secret, demo 1 query đúng + 1 out-of-domain + A/B.