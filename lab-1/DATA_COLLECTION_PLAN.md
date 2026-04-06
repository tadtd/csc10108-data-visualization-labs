# KẾ HOẠCH THU THẬP VÀ XÂY DỰNG DỮ LIỆU (DATA COLLECTION PLAN)

## 1. Tổng quan

### 1.1 Giới thiệu bài toán
Dự án tập trung vào việc phân tích các yếu tố ảnh hưởng đến hiệu quả kinh doanh của ngành hàng sách trên nền tảng thương mại điện tử Tiki. Việc hiểu rõ các biến số như giá cả, chương trình khuyến mãi, uy tín tác giả và phản hồi từ khách hàng sẽ giúp đưa ra các thông tin chi tiết về xu hướng thị trường và hành vi người tiêu dùng.

### 1.2 Mục tiêu của phase Thu thập & Xây dựng dữ liệu
* Thiết kế và triển khai hệ thống crawling dữ liệu tự động từ Tiki.
* Xây dựng tập dữ liệu (dataset) sạch, có cấu trúc, đạt quy mô tối thiểu 5000 bản ghi.
* Thực hiện Feature Engineering sơ bộ để chuẩn bị cho các mô hình phân tích và dashboard trực quan hóa.

### 1.3 Liên kết với yêu cầu Lab
Kế hoạch này tuân thủ nghiêm ngặt các yêu cầu trong Lab 01 về quy mô dữ liệu, tính đa dạng của các trường thông tin và quy trình xử lý dữ liệu khoa học.

## 2. Định nghĩa bài toán dữ liệu

### 2.1 Bài toán chung
Phân tích các yếu tố ảnh hưởng đến hiệu quả bán hàng (doanh số/số lượng bán) của các sản phẩm sách trên Tiki.

### 2.2 Các nhóm mục tiêu phân tích theo thành viên
* **Tuấn:** Tập trung vào ảnh hưởng của uy tín tác giả (Author Reputation) và các từ khóa tiêu đề (Title Keywords).
* **Kiệt:** Phân tích hiệu ứng FOMO (Ranking, Bestseller badge) và các giá trị gia tăng (Quà tặng, Phiên bản đặc biệt).
* **Đạt:** Nghiên cứu vai trò của Nhà xuất bản (Publisher), Thể loại (Category) và hình thức bán Combo.
* **Đàm Đạt:** Phân tích chiến lược giá (Pricing), tỷ lệ giảm giá (Discount) và mối tương quan với Rating/Review.
* **Tâm:** Phân tích nội dung mô tả (Description), đặc điểm vật lý (Số trang, Năm xuất bản) và sắc thái bình luận (Sentiment).

### 2.3 Mapping Dataset → Mục tiêu phân tích

| Thành viên | Mục tiêu SMART | Field cần dùng | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Tuấn** | Phân tích ảnh hưởng của uy tín tác giả và từ khóa tiêu đề đến doanh số trong 6 tháng. | `author`, `title`, `sold_count`, `total_books` | Chứng minh tác giả nổi tiếng có doanh số cao hơn. |
| **Kiệt** | Đánh giá tác động của hiệu ứng đám đông (Bestseller, Ranking) đến quyết định mua hàng. | `ranking`, `is_bestseller`, `is_top100`, `has_gift`, `sold_count` | Xác định các "badge" có thực sự thúc đẩy chuyển đổi. |
| **Đạt** | So sánh hiệu quả kinh doanh giữa các Nhà xuất bản và mô hình bán sách Combo. | `publisher`, `category`, `is_combo`, `sold_count` | Tìm ra NXB và hình thức đóng gói tối ưu. |
| **Đàm Đạt** | Tối ưu hóa chiến lược giá và khuyến mãi dựa trên phản hồi của khách hàng. | `price`, `original_price`, `discount_percent`, `rating`, `review_count`, `sold_count` | Tìm "điểm ngọt" về giá để đạt doanh số tối đa. |
| **Tâm** | Khai phá dữ liệu văn bản (mô tả, bình luận) và đặc điểm vật lý để dự báo mức độ hài lòng. | `description`, `content`, `page_count`, `publish_year`, `rating_score` | Sentiment analysis và phân tích đặc tính sản phẩm. |

**Chứng minh tính SMART:** Toàn bộ mục tiêu đều cụ thể (Specific), có thể đo lường qua các trường số (Measurable), khả thi với dữ liệu Tiki (Achievable), liên quan trực tiếp đến bài toán kinh doanh (Relevant) và có mốc thời gian snapshot (Time-bound).

## 3. Nguồn dữ liệu và phạm vi thu thập

### 3.1 Nguồn dữ liệu
* **Nền tảng:** Sàn thương mại điện tử Tiki (Tiki.vn).
* **Đối tượng:** Các sản phẩm thuộc ngành hàng Sách (Books).
* **Phương pháp:** Gọi trực tiếp API nội bộ của Tiki (ưu tiên) hoặc Crawling qua Web Scraping (BeautifulSoup/Selenium) 

### 3.2 Phạm vi dữ liệu
* **Số lượng:** Tối thiểu 5000 dòng (records).
* **Chiến lược:** Thu thập từ 5-10 danh mục chính (Kinh tế, Kỹ năng sống, Văn học, Thiếu nhi, Ngoại ngữ...). Mỗi danh mục lấy khoảng 1000 sản phẩm để đảm bảo tính đại diện.
* **Thời gian:** Dữ liệu thực tế tại thời điểm crawl (snapshot), kết hợp với dữ liệu lịch sử nếu có (số bán tích lũy).

### 3.3 Xử lý biến doanh số (sold_count)

Trong trường hợp API không trả về giá trị số chính xác cho `quantity_sold`, quy trình xử lý sẽ như sau:
* **Trường hợp dữ liệu dạng văn bản (e.g., "1000+", "5k+"):** Chuyển đổi về dạng số nguyên (1000, 5000) để tính toán.
* **Trường hợp thiếu dữ liệu trực tiếp:** Sử dụng các biến đại diện (Proxy variables) để ước lượng doanh số:
    * **review_count:** Tương quan thuận với số lượng bán (thường chiếm 2-5% tổng doanh số).
    * **ranking:** Thứ hạng càng thấp thì doanh số càng cao.
    * **badge (Bestseller):** Chỉ dấu cho các sản phẩm có doanh số top đầu.
* **Chuẩn hóa:** Toàn bộ dữ liệu doanh số sẽ được đưa về kiểu `Numeric` để phục vụ phân tích tương quan và hồi quy.

## 4. Thiết kế Dataset (Schema)

### 4.1 Bảng products (Bảng chính)

| Trường thông tin | Kiểu dữ liệu | Giải thích | Mapping mục tiêu |
| :--- | :--- | :--- | :--- |
| product_id | Integer | Mã định danh sản phẩm | Khóa chính |
| title | String | Tên sách | Tuấn, Kiệt, Tâm |
| price | Float | Giá bán hiện tại | Đàm Đạt |
| original_price | Float | Giá bìa (chưa giảm) | Đàm Đạt |
| discount_percent | Integer | % Giảm giá | Đàm Đạt |
| sold_count | Integer | Số lượng đã bán | Tất cả (Biến mục tiêu) |
| ranking | Integer | Thứ hạng trong danh mục | Kiệt |
| rating | Float | Điểm đánh giá trung bình | Đàm Đạt, Tâm |
| review_count | Integer | Tổng số lượt đánh giá | Đàm Đạt |
| author | String | Tên tác giả | Tuấn |
| publisher | String | Nhà xuất bản | Đạt |
| category | String | Danh mục sản phẩm | Đạt |
| publish_year | Integer | Năm xuất bản | Tâm |
| page_count | Integer | Số trang | Tâm |
| description | Text | Mô tả chi tiết sản phẩm | Tâm |
| is_bestseller | Boolean | Nhãn Bán chạy | Kiệt |
| is_top100 | Boolean | Thuộc Top 100 danh mục | Kiệt |
| is_combo | Boolean | Sản phẩm là dạng combo | Đạt |
| has_gift | Boolean | Có quà tặng kèm | Kiệt |

### 4.2 Bảng reviews 
* `review_id`: ID đánh giá.
* `product_id`: Liên kết với bảng products.
* `rating_score`: Điểm đánh giá cá nhân.
* `content`: Nội dung bình luận.
* `created_at`: Thời gian đánh giá.

### 4.3 Bảng authors 
Đây là bảng dữ liệu phái sinh (derived table) được tính toán từ bảng products để đánh giá uy tín tác giả:
* `author_name`: Tên tác giả (Khóa chính).
* `total_books`: Tổng số đầu sách của tác giả có trên sàn (count product_id).
* `total_sales`: Tổng doanh số tích lũy của tác giả (sum sold_count).
* `average_rating`: Điểm đánh giá trung bình các tác phẩm của tác giả.

### 4.4 Quan hệ giữa các bảng

* **products (1) — (N) reviews:** Một sản phẩm có thể có nhiều lượt đánh giá. Liên kết qua `product_id`.
* **authors (1) — (N) products:** Một tác giả có thể có nhiều đầu sách. Liên kết qua `author` (trong products) và `author_name` (trong authors).
* **Khóa chính (Primary Key):** `product_id` (bảng products), `review_id` (bảng reviews), `author_name` (bảng authors).
* **Khóa ngoại (Foreign Key):** `product_id` trong bảng reviews trỏ đến `product_id` trong bảng products.
* **Cơ chế Join:** Khi phân tích, các bảng sẽ được join theo `product_id` để kết hợp đặc điểm sản phẩm với phản hồi khách hàng, hoặc theo tên tác giả để lấy thông tin tổng quan về uy tín.

## 5. Quy trình thu thập dữ liệu (Data Crawling Pipeline)

* **Bước 1: Crawl danh sách sản phẩm (Listing):** Truy cập vào các URL category, lấy danh sách `product_id` và `product_url`.
* **Bước 2: Crawl chi tiết sản phẩm (Detail):** Truy cập từng URL sản phẩm hoặc gọi API `v2/products/{id}` để lấy toàn bộ thông tin trong Schema đã thiết kế.
* **Bước 3: Crawl review (Nếu có):** Gọi API `v2/reviews` với `product_id` tương ứng để lấy dữ liệu bình luận.
* **Bước 4: Tối ưu hóa:** Sử dụng `asyncio` hoặc `threading` để tăng tốc độ crawl; triển khai batching để lưu trữ dữ liệu trung gian, tránh mất dữ liệu khi gặp lỗi mạng.

### 5.2 Xử lý lỗi và chống bị chặn (Anti-bot handling)

Để đảm bảo quá trình thu thập diễn ra liên tục và không bị hệ thống Tiki chặn (block IP):
* **Retry request:** Tự động thử lại tối thiểu 3 lần khi gặp lỗi kết nối hoặc timeout.
* **Sleep random:** Thêm khoảng nghỉ ngẫu nhiên từ 1–3 giây giữa các yêu cầu để mô phỏng hành vi người dùng thật.
* **User-Agent rotation:** Thay đổi định danh trình duyệt (User-Agent) liên tục trong danh sách các trình duyệt phổ biến.
* **Logging:** Ghi lại nhật ký lỗi (error logs) chi tiết cho từng ID sản phẩm thất bại để có phương án crawl bù.
* **Timeout handling:** Thiết lập thời gian chờ (timeout) tối đa 10-20 giây cho mỗi request để tránh treo hệ thống.

## 6. Output Dataset

* **Định dạng file:** `.csv` (dùng cho phân tích nhanh) và `.json` (lưu trữ cấu trúc lồng nhau).
* **Cấu trúc thư mục:**

```text
project-root/
├── data/
│   ├── raw/                # Dữ liệu thô vừa crawl về
│   ├── processed/          # Dữ liệu đã qua xử lý và cleaning
└── scripts/
    ├── crawler.py          # Script thu thập dữ liệu
    └── preprocessing.py     # Script làm sạch và feature engineering
```

* **Quy ước đặt tên:** `tiki_books_[category].csv`

## 7. Kiểm tra chất lượng dữ liệu (Data Quality Check)

Sau khi thu thập, dataset cần đạt các tiêu chuẩn sau để đảm bảo độ tin cậy cho phân tích:
* **Quy mô:** Tổng số dòng dữ liệu thu được ≥ 5000 bản ghi.
* **Tính toàn vẹn (Integrity):** Không có giá trị trống (null) ở các trường quan trọng: `product_id`, `title`, `price`, `sold_count`.
* **Tính duy nhất (Uniqueness):** Không tồn tại các dòng dữ liệu trùng lặp hoàn toàn hoặc trùng `product_id`.
* **Phân bố dữ liệu (Distribution):** Các biến như `price`, `rating` phải có phân bố hợp lý, không bị lệch quá mức do lỗi thu thập.
* **Tính nhất quán (Consistency):** Dữ liệu giữa bảng `products` và `reviews` phải khớp nhau về ID; tổng `total_books` trong bảng `authors` phải khớp với số lượng thực tế trong bảng `products`.
