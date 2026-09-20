"""Hiệu chỉnh SCORE_THRESHOLD cho fallback của Task 9 (owner: Nguyễn Cảnh Duy).

Cách làm: chạy dense search (cosine gốc, KHÔNG phải RRF score) trên hai nhóm
query — in-domain (nằm trong corpus du lịch/luật du lịch) và out-of-domain
(ngoài corpus) — rồi chọn ngưỡng tách được hai phân phối.

    python -m scripts.calibrate_threshold

Ngưỡng chọn ra được điền vào SCORE_THRESHOLD trong .env và ghi lại ở
group_project/evaluation/RESULT.md phần calibration.
"""

import json
import statistics
import sys
from pathlib import Path

# Console Windows mặc định cp1252, không in được tiếng Việt có dấu.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task5_semantic_search import semantic_search  # noqa: E402


IN_DOMAIN = [
    "Khách du lịch có những quyền gì theo Luật Du lịch 2017?",
    "Điều kiện kinh doanh dịch vụ lữ hành quốc tế là gì?",
    "Hướng dẫn viên du lịch cần đáp ứng điều kiện nào để được cấp thẻ?",
    "Mức ký quỹ kinh doanh dịch vụ lữ hành nội địa là bao nhiêu?",
    "Khách quốc tế đến Việt Nam 8 tháng đầu năm đạt bao nhiêu lượt?",
    "Cơ sở lưu trú du lịch gồm những loại nào?",
    "Nghĩa vụ của khách du lịch được quy định ra sao?",
    "Chính sách phát triển du lịch của Nhà nước gồm những nội dung gì?",
]

OUT_OF_DOMAIN = [
    "Cách nấu phở bò gia truyền Hà Nội",
    "Giá cổ phiếu Apple hôm nay là bao nhiêu?",
    "How do I train a convolutional neural network in PyTorch?",
    "Lịch thi đấu vòng loại World Cup 2026 khu vực châu Âu",
    "Triệu chứng của bệnh tiểu đường tuýp 2",
    "Thủ tục đăng ký kết hôn với người nước ngoài",
    "Cấu hình Nginx reverse proxy cho ứng dụng Node.js",
    "Công thức tính diện tích hình cầu trong không gian",
]

REPORT_PATH = Path(__file__).parent.parent / "reports" / "threshold_calibration.json"


def best_dense_score(query: str) -> float:
    """Lấy cosine score gốc cao nhất của dense search cho một query."""
    results = semantic_search(query, top_k=5)
    return results[0]["score"] if results else 0.0


def summarize(label: str, scores: list[float]) -> dict:
    """Tóm tắt phân phối score của một nhóm query."""
    return {
        "group": label,
        "n": len(scores),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(statistics.fmean(scores), 4),
        "median": round(statistics.median(scores), 4),
    }


def main() -> None:
    in_scores = [best_dense_score(query) for query in IN_DOMAIN]
    out_scores = [best_dense_score(query) for query in OUT_OF_DOMAIN]

    in_summary = summarize("in-domain", in_scores)
    out_summary = summarize("out-of-domain", out_scores)

    # Ngưỡng nằm giữa đáy in-domain và đỉnh out-of-domain. Nếu hai phân phối
    # chồng lấn, lùi về trung điểm của median để giảm false fallback.
    low_in, high_out = in_summary["min"], out_summary["max"]
    if low_in > high_out:
        threshold = round((low_in + high_out) / 2, 3)
        separable = True
    else:
        threshold = round((in_summary["median"] + out_summary["median"]) / 2, 3)
        separable = False

    false_fallback = sum(1 for score in in_scores if score < threshold)
    missed_fallback = sum(1 for score in out_scores if score >= threshold)

    report = {
        "in_domain": in_summary,
        "out_of_domain": out_summary,
        "in_domain_scores": [round(score, 4) for score in in_scores],
        "out_of_domain_scores": [round(score, 4) for score in out_scores],
        "separable": separable,
        "recommended_threshold": threshold,
        "false_fallback_on_in_domain": false_fallback,
        "missed_fallback_on_out_of_domain": missed_fallback,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nĐã ghi {REPORT_PATH}")
    print(f"Đặt SCORE_THRESHOLD={threshold} trong .env")


if __name__ == "__main__":
    main()
