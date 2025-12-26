from pyspark.sql.functions import *
import pandas as pd
import os


def descriptive_statistics(df):
    """
    In thống kê mô tả sơ bộ ra màn hình console.
    """
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS (PREVIEW)")
    print("=" * 70)

    # 1. Chọn vài cột quan trọng để hiển thị demo (tránh in quá nhiều loạn mắt)
    demo_cols = ["Severity", "Temperature(F)", "Visibility(mi)", "Distance(mi)"]
    # Chỉ chọn những cột thực sự có trong df
    existing_cols = [c for c in demo_cols if c in df.columns]

    if existing_cols:
        df.select(existing_cols).describe().show()
    else:
        # Nếu không tìm thấy cột quen thuộc, hiển thị 5 cột đầu tiên
        df.select(df.columns[:5]).describe().show()

    # 2. Phân phối Severity (Mức độ nghiêm trọng)
    print("\nSEVERITY DISTRIBUTION:")
    if "Severity" in df.columns:
        df.groupBy("Severity").count().orderBy("Severity").show()
    elif "label" in df.columns:
        print("(Không thấy cột Severity, hiển thị cột label)")
        df.groupBy("label").count().orderBy("label").show()

    print("=" * 70)


def save_descriptive_to_file(df):
    """
    Tính toán chi tiết và lưu file CSV.
    Đã tối ưu hóa để chạy nhanh hơn trên Spark.
    """
    # ===== PATH =====
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    RESULT_DIR = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ===== 1. XÁC ĐỊNH CỘT SỐ =====
    # Lấy tất cả cột số trừ ID và các cột đã mã hóa (_idx)
    numeric_cols = [
        c for c, t in df.dtypes
        if t in ("int", "double", "float", "long")
           and c not in ["ID", "Severity", "label"]
           and not c.endswith("_idx")
    ]

    print(f"📊 Đang tính toán thống kê cho {len(numeric_cols)} cột số...")
    summary = []

    for c in numeric_cols:
        try:
            # --- TỐI ƯU HÓA: Tính Mean, Min, Max, Std... trong 1 lệnh Spark duy nhất ---
            stats = df.select(
                count(c).alias("Count"),
                mean(c).alias("Mean"),
                stddev(c).alias("Std"),
                min(c).alias("Min"),
                max(c).alias("Max"),
                skewness(c).alias("Skewness"),
                kurtosis(c).alias("Kurtosis")
            ).first()

            # Tính Quantile (riêng thằng này phải tính riêng)
            q1, median, q3 = df.approxQuantile(c, [0.25, 0.5, 0.75], 0.01)

            row = {
                "Feature": c,
                "Count": stats["Count"],
                "Mean": stats["Mean"],
                "Median": median,
                "Std": stats["Std"],
                # "Variance": stats["Std"]**2 if stats["Std"] else 0, # Có thể bỏ qua Variance nếu không cần thiết
                "Min": stats["Min"],
                "Max": stats["Max"],
                "Q1": q1,
                "Q3": q3,
                "Skewness": stats["Skewness"],
                "Kurtosis": stats["Kurtosis"]
            }
            summary.append(row)
        except Exception as e:
            print(f"⚠️ Không thể tính toán cột {c}: {e}")

    # ===== 2. LƯU FILE THỐNG KÊ CHI TIẾT =====
    if summary:
        pd.DataFrame(summary).to_csv(
            os.path.join(RESULT_DIR, "descriptive_statistics.csv"),
            index=False
        )

    # ===== 3. LƯU FILE PHÂN PHỐI SEVERITY =====
    target_col = "Severity" if "Severity" in df.columns else "label"

    if target_col in df.columns:
        dist_df = df.groupBy(target_col).count().orderBy(target_col)
        # Chuyển sang Pandas để lưu CSV
        dist_df.toPandas().to_csv(
            os.path.join(RESULT_DIR, "severity_distribution.csv"),
            index=False
        )

    print("✅ Đã lưu thống kê mô tả:")
    print(f"   - {os.path.join(RESULT_DIR, 'descriptive_statistics.csv')}")
    print(f"   - {os.path.join(RESULT_DIR, 'severity_distribution.csv')}")