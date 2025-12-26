# US Traffic Accidents Analysis & Prediction 🇺🇸

## 📌 Giới thiệu
Dự án tập trung vào **phân tích và dự đoán tai nạn giao thông tại Mỹ**
dựa trên bộ dữ liệu **US Accidents (2016–2023)**.

Dự án sử dụng:
- Apache Spark để xử lý dữ liệu lớn
- Python & Machine Learning (Logistic Regression)
- Trực quan hóa và phân tích thống kê

Mục tiêu:
- Phân tích các yếu tố ảnh hưởng đến tai nạn giao thông
- Tiền xử lý dữ liệu quy mô lớn bằng Spark
- Xây dựng mô hình dự đoán
- Đảm bảo chuẩn best practice cho dự án Big Data / ML

---

## 📊 Dataset
Dataset **US Accidents** có dung lượng lớn nên **KHÔNG được đẩy lên GitHub**.

🔗 Nguồn dữ liệu:
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

### Cách sử dụng dữ liệu
1. Tải dataset từ Kaggle
2. Giải nén và đặt vào:data/US_Accidents_March23.csv

---

## 🧠 Xử lý dữ liệu với Spark
Dữ liệu được tiền xử lý bằng **Apache Spark** để:
- Làm sạch dữ liệu
- Chọn lọc đặc trưng
- Chuẩn hóa dữ liệu cho mô hình ML

### 📂 Thư mục `data/`
Thư mục `data/` **chỉ chứa metadata nhỏ**, không chứa dữ liệu lớn.

