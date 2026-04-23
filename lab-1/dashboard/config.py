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
    "Giá, giảm giá và nhà xuất bản",
    "Huy hiệu bán chạy và quà tặng",
    "Thể loại, nhà xuất bản và combo",
    "Thông tin sách và phản hồi người mua",
    "Từ khóa tiêu đề và tác giả",
  ],
}