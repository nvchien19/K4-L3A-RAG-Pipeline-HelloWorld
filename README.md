# Day 8 — RAG Pipeline

## Mục tiêu

Mỗi nhóm xây dựng một chatbot RAG trả lời câu hỏi từ bộ tài liệu do nhóm thu thập. Sản phẩm phải có hybrid retrieval, citation, giao diện chat và báo cáo đánh giá.

Nhóm tự chọn bài toán và thu thập dữ liệu phù hợp; repo không cung cấp dữ liệu mẫu.

## Sản phẩm phải nộp

- Repository nhóm chạy được.
- Tối thiểu 3 tài liệu chính sách và 5 bài viết/page do nhóm tự thu thập.
- Pipeline: convert → chunk → index → dense + BM25 → RRF → fallback → generation có citation.
- Chatbot Streamlit hiển thị câu trả lời và nguồn đã dùng.
- Golden dataset tối thiểu 15 câu; đánh giá 4 metric và so sánh A/B.
- `group_project/evaluation/RESULT.md`.
- Mỗi thành viên nộp báo cáo cá nhân theo template trong `group_project/ịndividual/INDIVIDUAL_REPORT.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Điền API key cần dùng trong `.env`; không commit file này.

```bash
# 1. Thu thập và chuẩn hoá
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Làm sạch boilerplate của web nguồn (bắt buộc sau khi crawl lại)
python scripts/clean_standardized.py --dry-run   # xem trước sẽ cắt gì
python scripts/clean_standardized.py

# 3. Index và kiểm tra contract
python -m src.task4_chunking_indexing
pytest -q

# 4. Hiệu chỉnh ngưỡng fallback rồi ghi kết quả vào .env
python scripts/calibrate_threshold.py

# 5. Chạy sản phẩm
streamlit run app.py
```

> **Thứ tự các bước quan trọng.** `clean_standardized.py` phải chạy trước khi
> index, còn `calibrate_threshold.py` phải chạy sau khi index. Mỗi lần đổi
> corpus hoặc đổi embedding model thì `SCORE_THRESHOLD` cũ đều hết hiệu lực và
> phải đo lại, vì ngưỡng được so với cosine score gốc của dense search.

Kiểm tra embedding backend nào đang thực sự chạy (tránh trường hợp tưởng là
`BAAI/bge-m3` nhưng thực tế rơi vào hash fallback vì thiếu thư viện):

```bash
python -c "import src.task4_chunking_indexing as t4; t4.embed_texts(['test']); print(t4.describe_embedding_backend())"
```

Cấu hình mặc định dùng `EMBEDDING_PROVIDER=gemini` với `gemini-embedding-001`
(3072 chiều), gọi qua API nên không phải tải model về máy. Đổi sang chạy offline
bằng cách đặt `EMBEDDING_PROVIDER=sentence_transformers` và
`EMBEDDING_MODEL=BAAI/bge-m3`; lần đầu sẽ tải ~2.2GB và trên CPU không có GPU thì
index toàn bộ corpus mất rất lâu.

> **Đổi embedding provider thì phải index lại từ đầu.** Vector của hai model khác
> số chiều nhau (Gemini 3072, bge-m3 1024) nên không dùng chung collection được:
> xoá `chroma_db/` trước khi chạy lại bước 3, rồi chạy lại bước 4.

Free tier của Gemini giới hạn 100 request embed mỗi phút. Code tự đợi và thử lại
theo `retryDelay` server trả về thay vì rơi xuống hash fallback, nên nếu chạm
trần thì bước index chỉ chậm đi chứ không sinh index rác.

## Lộ trình 3 giờ

| Mốc                  | Thời gian | Kết quả cần có                           |
| -------------------- | --------: | ---------------------------------------- |
| 0. Setup             |   10 phút | Môi trường và `.env` sẵn sàng            |
| 1. Data              |   25 phút | ≥3 legal, ≥5 news, Markdown đã chuẩn hoá |
| 2. Index & search    |   30 phút | ChromaDB, dense search và BM25 chạy được |
| 3. Fusion & fallback |   25 phút | RRF và fallback tuân thủ contract        |
| 4. Generation & UI   |   30 phút | Chatbot trả lời có citation              |
| 5. Evaluation        |   30 phút | 15+ Q&A, 4 metric, A/B comparison        |
| 6. Demo & handoff    |   30 phút | Test, report, demo và push repository    |

## Lưu ý quy tắc để có code quality tốt:

- Dense và BM25 nên cùng trả về `SearchResult` theo một schema.
- RRF chỉ nên dùng để gộp thứ hạng và chỉ chạy một lần.
- Fallback dùng cosine score gốc của dense retrieval.
- Threshold phải được hiệu chỉnh trên query in domain và out of domain, không có một con số đúng cho mọi corpus.

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md): schema, interface và invariant mà code/test nên tuân theo.
- [Step-by-step guide](docs/STEP_BY_STEP.md): thứ tự triển khai và tiêu chí hoàn thành từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md): Rubric thang điểm.
- [Individual report](group_project/ịndividual/INDIVIDUAL_REPORT.md): template báo cáo cá nhân.
- [Suggested topics](docs/SUGGESTED_TOPICS.md): danh sách chủ đề tham khảo, không bắt buộc.

## Kiểm tra

```bash
# Contract tests
pytest tests/test_contracts.py -q

# Acceptance tests
pytest tests/test_acceptance.py -q

# Toàn bộ
pytest -q
```
