
# AccidentLogisticRegression
📘 README.md — Shopping Behavior Prediction with Logistic Regression (Apache Spark MLlib)
📌 Giới thiệu

Dự án này thực hiện phân tích và dự báo hành vi mua sắm theo từng quý của khách hàng dựa trên dữ liệu giao dịch thương mại điện tử.
Mô hình sử dụng:

 - Apache Spark MLlib để xử lý dữ liệu lớn và huấn luyện mô hình.

 - Logistic Regression để dự báo khả năng khách hàng tiếp tục mua hàng trong quý tiếp theo (label 0/1).

 - Kết hợp RFM analysis, đặc trưng theo quý, xu hướng mua sắm và nhiều feature nâng cao.

Dự án bao gồm:

 - Tiền xử lý dữ liệu (DataPreprocessing.py)

 - Thống kê mô tả (DescriptiveAnalysis.py)

 - Huấn luyện mô hình Logistic Regression (ModelLogisticRegression.py)

 - Đánh giá mô hình (ModelEvaluation.py)

 - Vẽ biểu đồ ROC / Feature importance (Chart.py)

 - Chương trình chính chạy pipeline (MainModel.py)

Ngoài phần xử lý và huấn luyện, hệ thống được tích hợp Flask Web Dashboard giúp người dùng:

 - Upload dữ liệu giao dịch .csv

 - Xem thống kê mô tả

 - Huấn luyện và đánh giá mô hình

 - Giải thích trọng số đặc trưng

 - Dự đoán hành vi cho khách hàng mới

📂 Cấu trúc thư mục
    Project/
    │
    ├── src/
    │   ├── app.py                    # Flask main app
    │   ├── src/views/
    │   │        ├── layout.html               # Layout chung
    │   │        ├── upload.html               # Trang upload dữ liệu
    │   │        ├── descriptive.html          # Trang thống kê mô tả
    │   │        ├── evaluation.html           # Trang đánh giá mô hình
    │   │        ├── explain.html              # Trang giải thích mô hình
    │   │        └── predict.html              # Trang dự đoán khách hàng mới
    │   ├── DataPreprocessing.py               # Tiền xử lý dữ liệu
    │   ├── DescriptiveAnalysis.py             # Thống kê mô tả
    │   ├── ModelLogisticRegression.py         # Huấn luyện mô hình
    │   ├── ModelEvaluation.py                 # Đánh giá mô hình
    │   ├── Chart.py                           # Vẽ biểu đồ ROC / Feature importance
    │   └── MainModel.py                       # Pipeline chính (kết nối toàn bộ)
    │
    ├── data/
    │   ├── data.csv                  # Dữ liệu đầu vào
    │   └── data_final.csv            # Sau khi tiền xử lý
    │
    ├── models/
    │   └── logistic_model.pkl        # Mô hình huấn luyện
    │
    ├── results/
    │   ├── descriptive_statistics.csv
    │   ├── label_distribution.csv
    │   ├── model_evaluation.csv
    │   ├── confusion_matrix.csv
    │
    ├── images/
    │   ├── roc_curve.png
    │   ├── label_distribution.png
    │   └── feature_importance.png
    │
    │
    └── README.md


🧱 1. Tiền xử lý dữ liệu
File: DataPreprocessing.py

Các bước chính:

    ✔ Làm sạch dữ liệu

        - Loại bỏ dòng thiếu Invoice, StockCode, Quantity, Price, InvoiceDate

        - Bỏ giao dịch Quantity ≤ 0 và Price ≤ 0

        - Chuyển InvoiceDate → timestamp

    ✔ Tạo các feature:

        - RFM & nâng cao: Recency, Frequency, Monetary, CustomerLifeSpan, AvgOrderValue…

        - Theo quý (Quarter-based): QuarterAmount, QuarterFrequency, QuarterAvgValue

        - Xu hướng hành vi (Trend): PurchaseTrend3Q (tăng, giảm hay không đổi so với quý trước)

    ✔ Gán nhãn (label)

        - Label = 1 nếu khách hàng có mua trong quý tiếp theo

        - Label = 0 nếu không có giao dịch tiếp theo

    ✔ Xuất dữ liệu cuối cùng

        - Dữ liệu được lưu tại: data/data_final.csv

📊 2. Thống kê mô tả
File: DescriptiveAnalysis.py

Bao gồm:

    ✔ Describe() cho tất cả biến số

    ✔ Phân phối nhãn (label distribution)

    ✔ Tính Quartiles (Q1, Median, Q3)

    ✔ Tính Skewness và Kurtosis

    ✔ Xuất file CSV vào thư mục results/

🤖 3. Huấn luyện Logistic Regression
File: ModelLogisticRegression.py

    ✔ Chuẩn hóa dữ liệu đúng chuẩn (Tránh leakage)

        - Chia train/test - 80/20 trước

        - Fit StandardScaler trên train

        - Transform test bằng scaler đã học

        - Chuẩn Z-score (withMean=True, withStd=True)

    ✔ Train Logistic Regression

    ✔ Lưu model

📈 4. Đánh giá mô hình
File: ModelEvaluation.py

Tất cả kết quả được lưu vào:

    ✔ results/model_evaluation.csv

    ✔ results/confusion_matrix.csv

📉 5. Biểu đồ trực quan
File: Chart.py

    ✔ ROC Curve

    ✔ Phân phối nhãn

    ✔ Độ quan trọng feature (Coefficient weights)

🚀 6. Chạy toàn bộ pipeline
File: MainModel.py

Pipeline sẽ tự động:

    ✔ Đọc data_final.csv

    ✔ Sinh thống kê mô tả vào results/

    ✔ Train Logistic Regression

    ✔ Đánh giá mô hình

    ✔ Vẽ biểu đồ ROC & Feature Importance

    ✔ Hoàn thành báo cáo

🧭 7. Web Dashboard
File: app.py + /views

Trang chính:

    ✔ upload.html: tải lên file CSV, tự động kích hoạt huấn luyện pipeline

    ✔ descriptive.html: xem thống kê mô tả + biểu đồ phân phối nhãn

    ✔ evaluation.html: hiển thị kết quả đánh giá mô hình (ROC, KPI, Confusion Matrix)

    ✔ explain.html: bảng hệ số và biểu đồ trọng số đặc trưng

    ✔ predict.html: form nhập dữ liệu khách hàng để dự đoán xác suất mua hàng quý tiếp theo

⚙️ 8. Cách chạy dự án

 - Cài đặt môi trường:
        pip install pyspark pandas numpy matplotlib flask

 - Chạy pipeline Spark:
        cd src
        python MainModel.py
    -> Kết quả sẽ sinh ra trong thư mục results/ và images/

 - Chạy Flask Web:
        cd src
        python app.py
    -> Mở trình duyệt tại: http://127.0.0.1:5000

📌 9. Yêu cầu hệ thống
    |   Thành phần    |   Phiên bản khuyến nghị  |
    | --------------- | ------------------------ |
    | Python          | 3.8 – 3.10               |
    | Apache Spark    | 3.x                      |
    | PySpark         | Tương thích Spark        |
    | Pandas          | ≥ 1.3                    |
    | Matplotlib      | ≥ 3.x                    |
    | NumPy           | ≥ 1.20                   |
    | Flask           | ≥ 2.x                    |
    | Bootstrap 5     | Giao diện web            |


❤️ 10. Liên hệ / Đóng góp