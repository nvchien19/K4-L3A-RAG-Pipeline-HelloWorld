"""Làm sạch data/standardized/**.md khỏi boilerplate của web nguồn.

Vì sao cần: file Markdown ở data/standardized được sinh ra bằng cách scrape
LuatVietnam.vn nên lẫn rất nhiều "chrome" của trang web (nút Theo dõi, menu điều
hướng, tường đăng nhập, danh sách văn bản liên quan). Đo thực tế trước khi sửa:
chuỗi "Đang theo dõi" lặp 857 lần, nhiễm 53.5% số chunk và chiếm 7.3% tổng ký tự.
Rác này đi thẳng vào embedding và BM25 index nên làm giảm chất lượng retrieval:
bỏ nó đi thì BM25-only recall@5 tăng từ 16/18 lên 17/18.

Script chạy được nhiều lần (idempotent) và chỉ đụng tới data/standardized.
File gốc ở data/landing không bị thay đổi.

Chạy:
    python scripts/clean_standardized.py            # làm sạch
    python scripts/clean_standardized.py --dry-run  # chỉ xem sẽ cắt bao nhiêu

LƯU Ý: sau khi chạy phải index lại rồi hiệu chỉnh lại ngưỡng:
    python -m src.task4_chunking_indexing
    python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
STANDARDIZED = ROOT / "data" / "standardized"

# Dòng chỉ chứa đúng một trong các chuỗi này là nút/nhãn giao diện, bỏ hẳn.
NOISE_LINES = {
    "Đang theo dõi",
    "Theo dõi",
    "Tải về",
    "Xem thêm",
    "Mục lục",
    "So sánh VB",
    "VB song ngữ",
    "VB gốc",
    "VB liên quan",
    "Lược đồ",
    "Tiếng Anh",
    "Hiệu lực",
    "Tổng quan",
    "Nội dung",
    "Nội dung hợp nhất",
    "Tiêu chuẩn",
    "Nâng cao",
    "hoặc",
    "Văn bản pháp luật",
    "Đây là tiện ích dành cho tài khoản",
    ". Vui lòng Đăng nhập tài khoản để xem chi tiết.",
    "Bạn chưa Đăng nhập thành viên.",
    "Vui lòng Đăng nhập để tải văn bản.",
    "Nếu chưa có tài khoản, vui lòng Đăng ký tại đây!",
    "văn bản tiếng việt",
    "văn bản TIẾNG ANH",
    "Công báo tiếng Anh",
}

# Dòng khớp các mẫu này cũng là chrome (quảng cáo tính năng, hướng dẫn dùng web).
NOISE_PATTERNS = [
    re.compile(r"^👉\s*Quay về:"),
    re.compile(r"^=&gt;&gt;\s*Xem hướng dẫn"),
    re.compile(r"^Tính năng này chỉ có tại LuatVietnam\.vn"),
    re.compile(r"^Khách hàng chỉ cần xem Nội dung hợp nhất"),
    re.compile(r"^Đây là tiện ích dành cho tài khoản"),
    re.compile(r"^Tiện ích dành cho tài khoản"),
    re.compile(r"^Vui lòng Đăng nhập"),
    re.compile(r"^\d{2}$"),  # số thứ tự "01".."07" trong danh sách VB liên quan
]

# Nội dung thật bắt đầu từ "Chương I" / "Điều 1." đầu tiên.
CONTENT_START = re.compile(r"^(Chương\s+I\b|Điều\s+1\.)")

# Mọi thứ từ các mốc này trở đi là tường đăng nhập + danh sách VB liên quan.
CONTENT_END = re.compile(
    r"^(Bạn chưa Đăng nhập thành viên\.|văn bản tiếng việt|văn bản TIẾNG ANH)"
)


def clean_text(text: str, *, is_legal: bool) -> str:
    """Trả về nội dung đã bỏ chrome. Giữ nguyên thứ tự và nội dung thật."""
    lines = text.splitlines()

    if is_legal:
        # Cắt phần đầu (menu điều hướng) — chỉ cắt khi tìm thấy mốc rõ ràng.
        for index, line in enumerate(lines):
            if CONTENT_START.match(line.strip()):
                lines = lines[index:]
                break
        # Cắt phần đuôi (tường đăng nhập, VB liên quan).
        for index, line in enumerate(lines):
            if CONTENT_END.match(line.strip()):
                lines = lines[:index]
                break

    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped in NOISE_LINES:
            continue
        if any(pattern.match(stripped) for pattern in NOISE_PATTERNS):
            continue
        kept.append(line)

    # Gộp các dòng trống liên tiếp thành một dòng trống.
    result = "\n".join(kept)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in thống kê, không ghi đè file.",
    )
    args = parser.parse_args()

    files = sorted(STANDARDIZED.rglob("*.md"))
    if not files:
        print(f"Không tìm thấy file .md nào trong {STANDARDIZED}")
        return 1

    total_before = total_after = 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        is_legal = path.parent.name == "legal"
        cleaned = clean_text(original, is_legal=is_legal)

        before, after = len(original), len(cleaned)
        total_before += before
        total_after += after
        saved = (1 - after / before) * 100 if before else 0.0

        status = "giữ nguyên" if before == after else f"-{saved:.1f}%"
        print(f"{path.relative_to(ROOT)}: {before:,} -> {after:,} ký tự ({status})")

        if not args.dry_run and cleaned != original:
            path.write_text(cleaned, encoding="utf-8")

    overall = (1 - total_after / total_before) * 100 if total_before else 0.0
    print(f"\nTổng: {total_before:,} -> {total_after:,} ký tự (giảm {overall:.1f}%)")
    if args.dry_run:
        print("(dry-run: chưa ghi file nào)")
    else:
        print("Đã ghi. Nhớ chạy lại:")
        print("  python -m src.task4_chunking_indexing")
        print("  python scripts/calibrate_threshold.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
