from numpy import sign
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import os


# Tạo Spark session
def create_spark():
    spark = SparkSession.builder \
        .appName("AccidentsLogisticRegression") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

# Load dữ liệu
def load_data(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    print("📥 Đã load dữ liệu:", df.count(), "dòng")
    return df

# Làm sạch dữ liệu
# 3. Làm sạch dữ liệu (Clean Data)
def clean_data(df):
    print("🧹 Đang làm sạch dữ liệu...")

    # Chọn các cột cần thiết cho MainModel
    # LƯU Ý: Phải giữ lại các cột mà MainModel.py cần dùng
    required_cols = [
        "Severity", "Start_Time", "Start_Lat", "Start_Lng",
        "Distance(mi)", "Temperature(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Weather_Condition", "Sunrise_Sunset"
    ]

    # Chỉ lấy các cột tồn tại trong file
    selected_cols = [c for c in required_cols if c in df.columns]
    df = df.select(selected_cols)

    # Loại bỏ dòng thiếu dữ liệu (Null) ở các cột quan trọng
    df = df.dropna(subset=selected_cols)

    # Loại bỏ các giá trị trùng lặp
    df = df.dropDuplicates()

    print(f"   Số dòng sau khi làm sạch: {df.count()}")
    return df


def feature_engineering(df):
    print("⚙️  Đang tạo feature thời gian (Hour, Weekday, Month)...")

    if "Start_Time" in df.columns:
        df = df.withColumn("Hour", hour("Start_Time")) \
            .withColumn("Weekday", dayofweek("Start_Time")) \
            .withColumn("Month", month("Start_Time"))

    return df
# 5. Mã hóa dữ liệu chữ (Hàm của bạn)
def encode_categorical_cols(df, cat_cols):
    """
    Phiên bản AN TOÀN: Tự động bỏ qua các cột không tìm thấy.
    """
    print(f"\n--- Đang mã hóa dữ liệu (Label Encoding) ---")

    # Lọc danh sách: Chỉ xử lý những cột THỰC SỰ CÓ trong df
    valid_cat_cols = [c for c in cat_cols if c in df.columns]

    if not valid_cat_cols:
        print("⚠️ Không có cột string nào hợp lệ để mã hóa.")
        return df, []

    indexers = []
    new_cat_cols = []

    for col_name in valid_cat_cols:
        new_col_name = col_name + "_idx"
        new_cat_cols.append(new_col_name)

        # handleInvalid="keep": Giữ lại giá trị lạ thay vì báo lỗi
        indexer = StringIndexer(inputCol=col_name, outputCol=new_col_name, handleInvalid="keep")
        indexers.append(indexer)

    try:
        pipeline = Pipeline(stages=indexers)
        model = pipeline.fit(df)
        encoded_df = model.transform(df)
        print(f"✅ Đã mã hóa thành công: {valid_cat_cols} -> {new_cat_cols}")
        return encoded_df, new_cat_cols

    except Exception as e:
        print(f"❌ Lỗi khi chạy StringIndexer: {e}")
        return df, []
# Save file output - Dùng Pandas thay vì Spark CSV
def save_output(df):
    output_folder = "../data"
    output_path = os.path.join(output_folder, "data_final_processed")
    print("Do dữ liệu quá lớn nên phải luư bằng Spark .Xin lỗi vì làm mất thời gian")
    print(f"💾 Đang lưu dữ liệu bằng Spark vào thư mục: {output_path}")
    # Tạo thư mục nếu chưa có
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        df.write.mode("overwrite").option("header", "true").csv(output_path)

        print(f"✅ LƯU THÀNH CÔNG!")
        print(f"   Lưu ý: Spark lưu thành một THƯ MỤC tên là '{os.path.basename(output_path)}'.")
        print(f"   Bên trong đó chứa các file .csv (part-0000...). Đây là tính năng của Spark.")

    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
# Pipeline chính
def preprocess(path):
    spark = create_spark()

    # 1. Load dữ liệu
    df = load_data(spark, path)

    # 2. Làm sạch (Thay vì feature_area/feature_time gây lỗi)
    df = clean_data(df)

    # 3. Tạo feature thời gian
    df = feature_engineering(df)

    # 4. Mã hóa Weather và Sunrise
    cat_cols = ["Weather_Condition", "Sunrise_Sunset"]
    df, encoded_cols = encode_categorical_cols(df, cat_cols)

    # In kết quả kiểm tra
    print("Pipeline hoàn tất.")
    # Lưu file
    save_output(df)
    print(f"Tổng số dòng dữ liệu sạch: {df.count()}")
    # Trả về DataFrame và danh sách cột mã hóa để MainModel dùng
    return df, encoded_cols

if __name__ == "__main__":
    # Đường dẫn file
    file_path = "../data/US_Accidents_March23.csv"

    if os.path.exists(file_path):
        df_result, cols_result = preprocess(file_path)
        df_result.show(5)
    else:
        print(f"❌ Không tìm thấy file tại: {file_path}")