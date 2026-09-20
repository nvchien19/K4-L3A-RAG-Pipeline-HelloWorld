# Kết quả đánh giá RAG

## 1. Thông tin chạy đánh giá

| Hạng mục | Giá trị |
| --- | --- |
| Ngày ghi nhận | 2026-09-20 |
| Corpus/commit | `da0e209` |
| Kích thước golden dataset | 18 câu hỏi có đáp án và ngữ cảnh chuẩn |
| `top_k` | 5 |
| Embedding | Cấu hình trong `.env`; mặc định theo `.env.example` là `BAAI/bge-m3` với `sentence_transformers` |
| Generator | `LLM_PROVIDER` và `LLM_MODEL` trong `.env` |
| RAGAS | Đã thử chạy `ragas==0.4.3`; chưa có kết quả hợp lệ vì credential Gemini bị API từ chối với HTTP 401 |

Calibration mới nhất trong `reports/threshold_calibration.json` được chạy trên 8 query in-domain và 8 query out-of-domain. Kết quả cho thấy điểm dense nằm trong khoảng `0.7878–0.9000` với nhóm in-domain và `0.5034–0.6715` với nhóm out-of-domain. Ngưỡng khuyến nghị hiện tại là `SCORE_THRESHOLD=0.73`, với 0 false fallback và 0 missed fallback trên mẫu calibration này. Ngưỡng cũ `0.424` không còn được dùng vì được đo trên trạng thái corpus trước đó.

## 2. Hai cấu hình được so sánh

- **Config A - dense-only:** chỉ dùng semantic search, tắt reranking và fallback bằng `score_threshold=-1.0`.
- **Config B - hybrid + RRF:** kết hợp dense search và BM25 bằng Reciprocal Rank Fusion; PageIndex chỉ được gọi khi điểm cosine dense thấp hơn ngưỡng.

Hai cấu hình dùng cùng golden dataset, prompt, generator và `top_k`; khác biệt duy nhất là chiến lược retrieval. Script chạy RAGAS là `scripts/evaluate_ragas.py`.

## 3. Điểm RAGAS

RAGAS đã được chạy và đã tạo `group_project/evaluation/ragas_scores.json`, nhưng không tạo ra kết quả hợp lệ. Generator/evaluator nhận lỗi HTTP 401 từ Gemini (`ACCESS_TOKEN_TYPE_UNSUPPORTED`), sau đó các lượt đánh giá bị lỗi hoặc nhận dữ liệu từ chối. Vì vậy các ô dưới đây vẫn để `chưa có kết quả hợp lệ`, không dùng các giá trị `NaN` hoặc `0.0` trong artifact để kết luận chất lượng.

| Metric | Config A | Config B | Chênh lệch B - A |
| --- | ---: | ---: | ---: |
| Faithfulness | chưa có kết quả hợp lệ | chưa có kết quả hợp lệ | chưa tính |
| Answer relevance | chưa có kết quả hợp lệ | chưa có kết quả hợp lệ | chưa tính |
| Context recall | chưa có kết quả hợp lệ | chưa có kết quả hợp lệ | chưa tính |
| Context precision | chưa có kết quả hợp lệ | chưa có kết quả hợp lệ | chưa tính |
| **Average** | chưa tính | chưa tính | chưa tính |

Lệnh chạy đầy đủ sau khi cài dependency và cấu hình API key:

```bash
python -m pip install -e .
python scripts/evaluate_ragas.py --top-k 5
```

Artifact đã được ghi vào `group_project/evaluation/ragas_scores.json`, nhưng cần chạy lại sau khi sửa credential. Chỉ sau một lần chạy không có lỗi provider mới nên kết luận cấu hình nào tốt hơn theo chất lượng câu trả lời.

## 4. A/B retrieval-only

Trong phép đo độc lập với LLM, metric là `recall@5` theo `source_document` trên 18 câu hỏi:

| Cấu hình | Kết quả |
| --- | ---: |
| Config A - dense-only | 17/18 (94,4%) |
| Config B - hybrid + RRF | 17/18 (94,4%) |

Hai cấu hình cùng bỏ sót `legal-09`. Đây là bằng chứng retrieval-only, không thay thế cho bốn metric RAGAS. Hai cấu hình vẫn có thể khác nhau về thứ hạng, độ sạch của context và chất lượng câu trả lời dù cùng đạt recall theo tài liệu.

Config A có chi phí và độ trễ thấp hơn. Config B tốn thêm BM25, bước RRF và đôi khi có fallback, nhưng phù hợp hơn với câu hỏi chứa thuật ngữ pháp lý, danh sách hoặc con số chính xác. Vì lần chạy RAGAS bị lỗi credential, báo cáo chưa kết luận Config B thắng về chất lượng đầu-cuối.

## 5. Các lỗi cần ưu tiên xử lý

| Câu hỏi | Kết quả hiện tại | Phân tích |
| --- | --- | --- |
| Có những loại cơ sở lưu trú du lịch nào? | Cả A và B cùng miss `legal-09` | Điều 48 chỉ nêu cụm “cơ sở lưu trú du lịch” một lần rồi liệt kê các mục ngắn. Những chunk khác lặp cụm từ này nhiều hơn nên dễ thắng cả dense search và BM25. Đây là lỗi retrieval dạng câu hỏi liệt kê. |
| Điều kiện kinh doanh lữ hành nội địa khác quốc tế thế nào về bằng cấp? | Chưa có điểm RAGAS theo từng cấu hình | Câu hỏi cần lấy đồng thời khoản 1 và khoản 2 của Điều 31 rồi so sánh “trung cấp” với “cao đẳng”. Nếu context bị cắt hoặc chỉ chứa một khoản, câu trả lời dễ thiếu vế. |
| Trong 8 tháng năm 2026, khách quốc tế đến bằng đường hàng không chiếm bao nhiêu? | Chưa có điểm RAGAS theo từng cấu hình | Đây là câu hỏi số liệu; context cần giữ cùng đoạn chứa `13,2 triệu`, `83,1%`, `15,5%` và `1,4%` để generator không nhầm mẫu số hoặc bỏ sót các phương thức còn lại. |

Artifact `reports/retrieval_ab_findings.json` ghi nhận phép đo A/B trên index cũ với 563 chunks và hash embedding fallback. Vì vậy, artifact này được dùng để mô tả phép đo retrieval đã có, không được hiểu là kết quả RAGAS hay benchmark cuối cùng sau mọi thay đổi corpus.

## 6. Kiểm tra acceptance và giới hạn

- Golden dataset có 18 case, vượt yêu cầu tối thiểu 15 case.
- Đã có script cho bốn metric bắt buộc: faithfulness, answer relevance, context recall và context precision.
- Đã có calibration cho fallback và file kết quả `reports/threshold_calibration.json`.
- UI citation/source highlighting đã được triển khai trong `app.py`, gồm nội dung nguồn, phương thức retrieval và score.
- RAGAS đã được cài và đã thử chạy, nhưng chưa có kết quả hợp lệ vì credential Gemini hiện tại bị từ chối HTTP 401. File `ragas_scores.json` hiện có các giá trị `NaN`/`0.0` phát sinh từ lần chạy lỗi và không được dùng làm điểm chính thức.
- Test đã chạy thành công trong virtual environment: `.venv\Scripts\python.exe -m pytest -q` → **27 passed**.

## 7. Việc cần làm tiếp

1. Thay `GEMINI_API_KEY` bằng Gemini API key hợp lệ từ Google AI Studio, không dùng OAuth access token dạng `AQ...`; đặt `LLM_PROVIDER=gemini` và chọn model còn được tài khoản hỗ trợ.
2. Chạy lại `scripts/evaluate_ragas.py --top-k 5` và kiểm tra cả hai config đều có answer/context hợp lệ, không còn `NaN` do lỗi provider.
3. Bổ sung score của hai cấu hình vào bảng trên, tính delta và cập nhật kết luận A/B.
4. Điều chỉnh chunking theo ranh giới Điều/khoản hoặc bổ sung chiến lược ưu tiên heading cho các câu hỏi liệt kê như `legal-09`.
5. Chạy lại index và `scripts/calibrate_threshold.py` sau mọi thay đổi corpus hoặc embedding; không giữ lại ngưỡng cũ một cách tự động.
6. Giữ kiểm tra hồi quy bằng `.venv\Scripts\python.exe -m pytest -q` trước khi demo.

## 8. Bonus

| Hạng mục | Trạng thái | Nhận xét |
| --- | --- | --- |
| UI citation/source highlighting | Đã triển khai | `app.py` hiển thị answer, nguồn, phương thức retrieval và score. Cần demo trực tiếp để hoàn tất bằng chứng bonus. |
