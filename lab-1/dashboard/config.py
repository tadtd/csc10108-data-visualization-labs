from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRODUCTS_PATH = PROJECT_ROOT / "data" / "processed" / "products_clean.csv"
AUTHORS_PATH = PROJECT_ROOT / "data" / "processed" / "authors_clean.csv"
REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv"

APP_TITLE = "Các yếu tố ảnh hưởng đến hiệu quả bán hàng của mặt hàng sách trên Tiki"

SIDEBAR_SETTINGS = {
  "title": "Điều hướng dashboard",
  "tab_label": "Chọn chuyên đề phân tích",
  "tabs": [
    "Giá & Nhà xuất bản",
    "FOMO & Quà tặng",
    "Chiến lược thể loại",
    "Thông tin & Đánh giá",
    "Từ khóa phổ biến",
  ],
}