from pyspark.sql import SparkSession
from DataPreprocessing import encode_categorical_cols
from DescriptiveAnalysis import descriptive_statistics, save_descriptive_to_file
from ModelLogisticRegression import train_model, save_model_pkl
from ModelEvaluation import evaluate_model, save_evaluation_to_file
from Chart import (plot_roc, plot_label_distribution, plot_feature_importance,
                   extract_roc_points, plot_feature_histogram, plot_boxplot, add_label)

import os


def main():
    # ---------- SPARK ----------
    spark = SparkSession.builder.appName("TrafficAccidentsModel").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")  # Gọn log khi chạy

    # ================= PATH =================
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
    BASE_DIR = os.path.dirname(SRC_DIR)  # project root

    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGE_DIR = os.path.join(BASE_DIR, "images")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # ---------- LOAD DATA ----------
    csv_path = os.path.join(DATA_DIR, "../data/data_final_processed")

    print(f"--- Đang đọc dữ liệu từ: {csv_path}")
    df = spark.read.csv(
        csv_path,
        header=True,
        inferSchema=True
    )

    # ---------- TIỀN XỬ LÝ QUAN TRỌNG ----------
    # Chuyển đổi Severity (1-4) thành label (0-1) để vẽ biểu đồ và train
    # 2. TẠO NHÃN (LABEL)
    # =========================================================
    print("--- Đang tạo nhãn (Label) từ Severity...")
    # Hàm add_label (từ Chart.py) chuyển Severity 1-4 thành label 0-1
    df = add_label(df)
    label_col = "label"

    # =========================================================
    # 3. CHỌN FEATURE (Cập nhật đầy đủ nhất)
    # =========================================================
    feature_cols = [
        "Start_Lat", "Start_Lng", "Distance(mi)", "Temperature(F)",
        "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Hour", "Month", "Weekday"  # <-- Thêm các feature thời gian vừa tạo được
    ]

    # Tổng hợp Feature = Cột số + Cột mã hóa (Weather/Sunrise)
    # ---------- 1. PHÂN TÍCH MÔ TẢ ----------
    try:
        descriptive_statistics(df)
        save_descriptive_to_file(df)
    except NameError:
        print("Skipping descriptive statistics (Module not found)")

    # Phân phối mức độ tai nạn
    print("--- Đang vẽ biểu đồ phân phối nhãn...")
    plot_label_distribution(
        df,
        save_path=os.path.join(IMAGE_DIR, "severity_distribution.png"),
        title="Phân phối mức độ tai nạn (0: Nhẹ, 1: Nghiêm trọng)"
    )

    # Vẽ histogram cho các feature số
    print("--- Đang vẽ Histogram...")
    for feature in ["Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)"]:

        if feature in df.columns:
            plot_feature_histogram(df, feature, save_path=os.path.join(IMAGE_DIR, f"{feature}_hist.png"))

    # Boxplot khoảng cách tai nạn
    if "Distance(mi)" in df.columns:
        plot_boxplot(df, "Distance(mi)", save_path=os.path.join(IMAGE_DIR, "distance_boxplot.png"))

    # ---------- 2. TRAIN MODEL ----------
    print(f"--- Đang huấn luyện mô hình với label='{label_col}'...")
    # Đảm bảo train_model nhận đúng cột label (0/1)
    model, predictions, scaler_model = train_model(df, feature_cols)

    # ---------- 3. LƯU MODEL ----------
    save_model_pkl(model, scaler_model, feature_cols)

    # ---------- 4. ĐÁNH GIÁ MODEL ----------
    print("--- Đang đánh giá mô hình...")

    metrics, cm = evaluate_model(predictions)
    save_evaluation_to_file(metrics, cm)

    # ---------- 5. ROC ----------
    print("--- Đang vẽ ROC Curve...")

    fpr, tpr = extract_roc_points(predictions)
    plot_roc(fpr, tpr, metrics["AUC-ROC"], save_path=os.path.join(IMAGE_DIR, "roc_curve.png"))

    # ---------- 6. FEATURE IMPORTANCE ----------
    plot_feature_importance(
        model.coefficients,
        feature_cols,
        save_path=os.path.join(IMAGE_DIR, "feature_importance.png")
    )

    print("\n✅ MÔ HÌNH DỰ ĐOÁN TAI NẠN HOÀN THÀNH!")
    spark.stop()


if __name__ == "__main__":
    main()