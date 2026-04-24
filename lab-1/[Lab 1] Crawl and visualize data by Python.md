# [Lab 1] Crawl and visualize data by Python

## Goal

- Collect real-world data
- Build an interactive dashboard
- Analyze and provide justification
- **EDA and Visualization**
- Dashboard phải cùng 1 tone màu (khái niệm về màu, mù màu, thêm option cho người mù màu)
- Không dùng scatter nếu cột x hoặc y là integer
- Pivot plot

## Define and analysis objectives

- E-commerce platforms: Tiki, Shopee, Lazada, Sendo,…
- **SMART** criteria
    - **S**pecific
    - **M**easurable
    - **A**chievable
    - **R**elevant
    - **T**ime-bound
- Problem: each groups define it themselves (?)

## Requirements

- Data collection, construction and preprocessing
    - Select one or more e-commerce platforms to collect data using web crawling or APIs
    - Design and build a well-structured database/dataset that clearly represents relationships between entities
    - The dataset must contain at least 5000 records
- Data analysis and visualization
    - EDA
    - Visualization (Must use a variety of chart types learned in class)
    - Comments and conclusions
    - Interactive dashboard: Summarize the analysis result in an interactive dashboard (using `python` ). The dashboard must have at least 3 tabs.
- Machine learning

## Notes

- Tech stack:
    - Language: `Python`
    - Dashboard: Streamlit, Dash, Panel
    - Visualization: Plotly, Matplotlib, Seaborn,…
- Report: [Crawl and visualize](https://prism.openai.com/?u=c86cde7f-3064-4ebb-b276-a0392deae5f0&pg=1&m=main.tex)

# Assignment hub

# Bài toán chung: Phân tích các yếu tố ảnh hưởng đến hiệu quả bán hàng của mặt hàng sách trên Tiki

## Tuấn

### **Mục tiêu 1: Ảnh hưởng của keyword và pattern tiêu đề sách đến doanh số**

Phân tích mối quan hệ giữa các đặc trưng **keyword và pattern trong tiêu đề sách** (sự xuất hiện của các từ khóa như “best seller”, “tái bản”, “combo”, “tâm lý”, “kỹ năng”, sự hiện diện của số, dấu phân cách như “:”, “-”, và cấu trúc liệt kê) với hiệu quả bán hàng (số lượng bán/doanh thu) trên Tiki trong tập dữ liệu thu thập, nhằm xác định **3 đặc trưng tiêu đề quan trọng nhất** giúp sách có doanh số trung bình cao hơn **ít nhất 20%** so với nhóm còn lại trước khi hoàn thành dashboard.

#### SMART:

- **S:** Keyword trong tiêu đề, pattern tiêu đề (có số, dấu “:”, “-”, dạng liệt kê), số lượng bán/doanh thu
- **M:** Xác định top 3 đặc trưng + chênh lệch ≥ 20%
- **A:** Thực hiện bằng text processing đơn giản (regex, keyword matching, feature extraction) & NLP đơn giản (tokenize, keyword count)
- **R:** Tiêu đề ảnh hưởng trực tiếp đến khả năng hiển thị và thu hút người dùng trên TMĐT
- **T:** Thực hiện trên dataset thu thập và hoàn thành trước khi xây dựng dashboard

---

### Mục tiêu 2: **Phân tích ảnh hưởng của độ phổ biến tác giả đến doanh số sách**

Phân tích mối quan hệ giữa mức độ phổ biến của tác giả (số lượng sách của tác giả trên Tiki, tổng lượt bán của tác giả) và hiệu quả bán hàng của từng cuốn sách trong 6 tháng gần nhất, nhằm xác định liệu các sách thuộc **Top 10 tác giả phổ biến** có doanh số trung bình cao hơn ít nhất **25%** so với các tác giả còn lại.

#### SMART:

- **S:** Tác giả, số sách/tác giả, tổng sales/tác giả, sales từng sách
- **M:** Top 10 tác giả + ≥25% chênh lệch
- **A:** Crawl/Gọi API để lấy tên tác giả + groupby rất dễ
- **R:** Rất phù hợp về ngành sách
- **T:** 6 tháng gần nhất

## Kiệt

## **Mục tiêu 1: Đánh giá hiệu ứng FOMO từ các danh hiệu/huy hiệu (Bestseller Badge / Top Ranking)**

**Nội dung:** Phân tích mức độ ảnh hưởng của việc sở hữu huy hiệu "Top Bán Chạy" (đứng trong Top 100 của một danh mục con) đến hiệu quả bán hàng tổng thể. Nhằm đánh giá định lượng xem hiệu ứng FOMO (Fear Of Missing Out) từ danh hiệu này giúp sản phẩm duy trì mức doanh số trung bình vượt trội hơn ít nhất 50% so với các sản phẩm cùng thể loại nhưng không lọt Top hay không.

**Giải thích theo SMART:**

- **S (Specific):** Trạng thái có/không có huy hiệu "Top Bán Chạy", danh mục con, số lượng bán ra.
- **M (Measurable):** So sánh 2 nhóm (có Top vs không có Top) và kiểm tra ngưỡng chênh lệch $\ge$ 50%.
- **A (Achievable):** Rất dễ thực hiện bằng cách lọc dữ liệu theo trường "Badge/Rank" (boolean) và tính toán các chỉ số thống kê mô tả (Mean, Median) kết hợp biểu đồ cột.
- **R (Relevant):** Huy hiệu "Top Bán Chạy" là một minh chứng mạnh mẽ cho chất lượng, kích thích tâm lý đám đông của người mua hàng trên Tiki.
- **T (Time-bound):** Thực hiện phân tích trong giai đoạn EDA trước khi tiến hành xây dựng dashboard cuối kỳ.

## **Mục tiêu 2: Đánh giá sức hút của các "Giá trị cộng thêm" (Bản đặc biệt, Quà tặng kèm)**

**Giải thích theo SMART:**

**Nội dung:** Phân tích tác động của các từ khóa chỉ định giá trị cộng thêm (như "Bản đặc biệt", "Tặng kèm Bookmark", "Postcard", "Kèm CD") trong tiêu đề hoặc mô tả đến hiệu quả bán hàng. Mục đích kiểm chứng xem nhóm sách có yếu tố sưu tầm/quà tặng có số lượng bán trung bình cao hơn ít nhất 20% so với bản tiêu chuẩn hay không.

- **S (Specific):** Từ khóa quà tặng/bản đặc biệt trong tiêu đề/mô tả, số lượng bán.
- **M (Measurable):** Tách thành 2 nhóm (Có quà tặng/Bản đặc biệt vs Bản thường) và đo lường mức chênh lệch $\ge$ 20%.
- **A (Achievable):** Dùng Regex hoặc String Matching đơn giản trên cột Tiêu đề/Mô tả để tạo cờ đánh dấu (Flag 0-1). Sau đó dùng Groupby và Bar chart để so sánh kết quả.
- **R (Relevant):** Trong cộng đồng mua bán sách online (đặc biệt mảng Văn học, Tiểu thuyết, Truyện tranh), các phụ kiện tặng kèm là thủ thuật kích cầu mạnh mẽ, tạo hiệu ứng sưu tầm và thôi thúc chốt đơn nhanh chóng.
- **T (Time-bound):** Xử lý văn bản và phân tích trên dữ liệu crawl trước khi chuyển sang vẽ biểu đồ tổng hợp.

# Đỗ Đạt

## Mục tiêu 1: Phân tích ảnh hưởng của thể loại sách và uy tín nhà xuất bản/tác giả đến doanh số

- **Nội dung:** Phân tích sự tác động của các nhóm thể loại (Kinh tế, Kỹ năng, Văn học, Thiếu nhi...) và uy tín của nhà xuất bản đến số lượng bán ra trên Tiki trong 6 tháng gần nhất, nhằm xác định 3 thể loại đang dẫn đầu thị trường và kiểm chứng xem các sản phẩm từ Top 5 nhà xuất bản lớn có doanh số trung bình cao hơn ít nhất 30% so với nhóm còn lại hay không.
- **Giải thích theo SMART:**
    - **S (Specific):** Thể loại sách, nhà xuất bản, số lượng bán.
    - **M (Measurable):** Xác định Top 3 thể loại; chênh lệch doanh số ≥ 30%.
    - **A (Achievable):** Phân loại dữ liệu theo danh mục và tên NXB; dùng Bar chart/Pie chart để so sánh.
    - **R (Relevant):** Giúp định vị phân khúc sách và đối tác cung cấp chiến lược.
    - **T (Time-bound):** Dữ liệu 6 tháng gần nhất.

## Mục tiêu 2: Phân tích hiệu quả của chiến lược bán theo Combo và định dạng sách (Bìa cứng/Bìa mềm)

- **Nội dung:** Phân tích sự khác biệt về doanh thu và tỷ lệ bán ra giữa hình thức bán lẻ và bán theo bộ (Combo) đối với các thể loại sách khác nhau trên Tiki trong quý gần nhất. Mục đích nhằm xác định 2 thể loại sách có hiệu quả kích cầu tốt nhất khi bán theo Combo (số bán tăng > 20% so với bán lẻ) để gợi ý chiến lược đóng gói sản phẩm tối ưu cho gian hàng.
- **Giải thích theo SMART:**
    - **S (Specific):** Hình thức bán (Lẻ vs Combo), thể loại sách, doanh thu, số lượng bán.
    - **M (Measurable):** Xác định 2 thể loại có mức tăng trưởng số bán > 20% khi chuyển sang Combo.
    - **A (Achievable):** Lọc từ khóa "Combo/Bộ" trong tên sản phẩm và gộp nhóm theo thể loại; dùng Grouped Bar Chart hoặc Pivot Plot để so sánh.
    - **R (Relevant):** Bán theo Combo là đặc thù của ngành sách giúp tăng giá trị đơn hàng (AOV) và giảm chi phí vận hành.
    - **T (Time-bound):** Dữ liệu trong quý gần nhất.

# Đàm Đạt

## Mục tiêu 1: Ảnh hưởng của giá–giảm giá–đánh giá đến doanh số sách

Phân tích trong 6 tháng gần nhất trên Tiki mối quan hệ giữa giá bán, % giảm giá, rating và số lượt review của sách với hiệu quả bán hàng (số lượng bán/doanh thu), nhằm xác định 3 yếu tố tác động mạnh nhất và kiểm tra liệu nhóm sách có rating ≥ 4.5 có số lượng bán trung bình cao hơn ít nhất 20% so với nhóm còn lại hay không.

Giải thích theo SMART:

- S (Specific): Tập trung vào sách trên Tiki; biến đầu vào gồm giá, % giảm giá, rating, số review; biến đầu ra gồm số lượng bán và/hoặc doanh thu.
- M (Measurable):
    - Xác định “Top 3” yếu tố ảnh hưởng mạnh nhất (theo correlation/regression feature importance).
    - So sánh chênh lệch số lượng bán trung bình giữa 2 nhóm rating ≥ 4.5 vs < 4.5 và kiểm tra ngưỡng ≥ 20%.
- A (Achievable): Dữ liệu có thể crawl từ trang sản phẩm; xử lý bằng EDA + groupby + correlation/regression cơ bản (Linear Regression/Random Forest).
- R (Relevant): Giá và đánh giá là 2 nhóm yếu tố quan trọng quyết định hành vi mua sách trên sàn TMĐT và trực tiếp liên quan đến doanh số.
- T (Time-bound): Giới hạn dữ liệu trong 6 tháng gần nhất; hoàn thành phân tích trước khi chốt dashboard/báo cáo cuối kỳ.

## Mục tiêu 2: Thể loại sách & uy tín nhà xuất bản ảnh hưởng thế nào đến doanh số

Phân tích trong quý gần nhất sự khác biệt về hiệu quả bán hàng của sách (số lượng bán/doanh thu) giữa các thể loại (Kinh tế, Kỹ năng, Văn học, Thiếu nhi, …) và giữa các nhóm nhà xuất bản (Top 5 NXB theo tổng doanh thu hoặc tổng số bán vs các NXB còn lại), nhằm xác định 3 thể loại dẫn đầu và kiểm chứng liệu sách thuộc Top 5 NXB có doanh số trung bình cao hơn ít nhất 30% so với nhóm còn lại hay không.

Giải thích theo SMART:

- S (Specific): Đối tượng là sách trên Tiki; so sánh theo 2 trục: thể loại sách và nhóm NXB (Top 5 vs còn lại); đo hiệu quả bằng số lượng bán/doanh thu.
- M (Measurable):
    - Xếp hạng và chọn Top 3 thể loại theo doanh thu hoặc số bán.
    - Đo chênh lệch doanh số trung bình Top 5 NXB vs còn lại và kiểm tra điều kiện ≥ 30% (có thể kèm kiểm định t-test).
- A (Achievable): Có thể trích xuất thể loại/NXB từ metadata hoặc từ breadcrumb/thuộc tính sản phẩm; phân tích bằng groupby, pivot, bar chart/boxplot và kiểm định thống kê cơ bản.
- R (Relevant): Thể loại phản ánh nhu cầu thị trường; NXB phản ánh uy tín/chất lượng, thường ảnh hưởng lớn đến quyết định mua sách và hiệu quả bán hàng.
- T (Time-bound): Giới hạn trong quý gần nhất; hoàn thành trước khi viết phần kết luận và hoàn thiện dashboard.

# Tâm

### **Mục tiêu 1: Phân tích ảnh hưởng của mức độ hoàn thiện thông tin sách đến hiệu quả bán hàng**

Phân tích mối quan hệ giữa **mức độ hoàn thiện thông tin của sản phẩm sách** trên Tiki (sự đầy đủ của mô tả nội dung, thông tin tác giả, nhà xuất bản, năm xuất bản, hình thức bìa, số trang và các thông tin giới thiệu liên quan) với hiệu quả bán hàng (số lượng bán/doanh thu) trong tập dữ liệu thu thập, nhằm xác định liệu các sách có **thông tin trình bày đầy đủ hơn** có doanh số trung bình cao hơn ít nhất **20%** so với nhóm còn lại, đồng thời xác định **3 thành phần thông tin quan trọng nhất** trước khi hoàn thành dashboard.

### SMART:

- **S:** Độ đầy đủ thông tin sách, metadata sản phẩm, số lượng bán/doanh thu
- **M:** So sánh chênh lệch doanh số trung bình ≥ 20% và xác định top 3 thành phần thông tin quan trọng nhất
- **A:** Có thể thực hiện bằng feature engineering từ dữ liệu crawl, thống kê mô tả và so sánh giữa các nhóm
- **R:** Với mặt hàng sách, người mua thường dựa nhiều vào thông tin chi tiết để đánh giá mức độ phù hợp trước khi quyết định mua
- **T:** Thực hiện trên tập dữ liệu thu thập và hoàn thành trước giai đoạn xây dựng dashboard

---

### **Mục tiêu 2: Phân tích ảnh hưởng của cảm nhận người mua qua review đến hiệu quả bán hàng của sách**

Phân tích mối quan hệ giữa **cảm nhận của người mua thể hiện qua review** (điểm đánh giá trung bình, mức độ tích cực của nội dung nhận xét, tần suất xuất hiện các từ khóa như “nội dung hay”, “giao nhanh”, “đẹp”, “đáng mua”, “sách mới”, “chất lượng tốt”) với hiệu quả bán hàng của sách trên Tiki trong tập dữ liệu thu thập, nhằm xác định liệu các sách có **review tích cực hơn** có doanh số trung bình cao hơn ít nhất **15%** so với nhóm còn lại và xác định **2 tín hiệu review quan trọng nhất**.

### SMART:

- **S:** Rating trung bình, nội dung review, từ khóa tích cực, số lượng bán/doanh thu
- **M:** Chênh lệch doanh số trung bình ≥ 15% và xác định 2 tín hiệu review quan trọng nhất
- **A:** Có thể thực hiện bằng thống kê, keyword analysis, sentiment analysis đơn giản hoặc rule-based
- **R:** Review là nguồn thông tin tham khảo rất quan trọng với người mua sách trên sàn thương mại điện tử
- **T:** Thực hiện trên tập dữ liệu thu thập và hoàn thành trước khi xây dựng dashboard