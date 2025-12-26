import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_curve, roc_auc_score
from pyspark.sql.functions import when, col

# --------- STYLE CHUNG CHO BIỂU ĐỒ ----------
def _apply_common_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10
    })


# --------- VẼ ROC CURVE ----------
def plot_roc(fpr, tpr, auc, save_path):
    _apply_common_style()
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="lower right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# --------- VẼ PHÂN BỐ NHÃN ----------
def plot_label_distribution(df, save_path, title="Label Distribution"):
    _apply_common_style()
    pdf = df.groupBy("label").count().toPandas().sort_values("label")

    plt.figure(figsize=(6.5, 4.8))
    plt.bar(pdf["label"].astype(str), pdf["count"], alpha=0.9, edgecolor="white")
    plt.xlabel("Tai nạn nghiêm trọng (0 / 1)")
    plt.ylabel("Số lượng")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# --------- VẼ FEATURE IMPORTANCE ----------
def plot_feature_importance(weights, feature_cols, save_path):
    _apply_common_style()
    weights = np.array(weights)
    order = np.argsort(np.abs(weights))
    sorted_features = np.array(feature_cols)[order]
    sorted_weights = weights[order]

    plt.figure(figsize=(9.5, 6.5))
    plt.barh(sorted_features, sorted_weights, alpha=0.9)
    plt.xlabel("Coefficient Weight")
    plt.title("Feature Importance - Logistic Regression")
    plt.grid(axis="x", linestyle="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# --------- TRÍCH XUẤT ROC POINTS ----------
def extract_roc_points(predictions):
    pdf = predictions.select("label", "probability").toPandas()
    y_true = pdf["label"].values
    y_score = np.array([p[1] for p in pdf["probability"]])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


# --------- VẼ HISTOGRAM CHO FEATURES ----------
def plot_feature_histogram(df, feature, save_path):
    _apply_common_style()
    pdf = df.select(feature).dropna().toPandas()

    plt.figure(figsize=(6.5, 4.8))
    plt.hist(pdf[feature], bins=30, alpha=0.88, edgecolor="white")
    plt.xlabel(feature)
    plt.ylabel("Tần suất")
    plt.title(f"Phân phối {feature}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# --------- VẼ BOXPLOT CHO FEATURES SỐ THEO NHÃN ----------
# Đã sửa tên hàm này từ plot_numeric_boxplot thành plot_boxplot
def plot_boxplot(df, feature, save_path):
    _apply_common_style()
    pdf = df.select(feature, "label").dropna().toPandas()

    data0 = pdf[pdf["label"] == 0][feature]
    data1 = pdf[pdf["label"] == 1][feature]

    plt.figure(figsize=(6.5, 4.8))
    plt.boxplot(
        [data0, data1],
        labels=["Không nghiêm trọng", "Nghiêm trọng"],
        patch_artist=True,
        boxprops=dict(alpha=0.75),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(alpha=0.8),
        capprops=dict(alpha=0.8)
    )
    plt.ylabel(feature)
    plt.title(f"So sánh {feature} theo mức độ nghiêm trọng")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# --------- TIỀN XỬ LÝ LABEL CHO DATASET TAI NẠN ---------
def add_label(df):
    """
    Severity >= 3 => label 1 (tai nạn nghiêm trọng)
    Severity < 3 => label 0 (tai nạn nhẹ)
    """
    # 1. Làm sạch tên cột (bỏ khoảng trắng thừa nếu có)
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.strip())

    # 2. Kiểm tra xem cột Severity có tồn tại không
    if "Severity" not in df.columns:
        print(f"❌ LỖI: Không tìm thấy cột 'Severity'. Các cột hiện có: {df.columns}")
        raise ValueError("Thiếu cột Severity trong dữ liệu!")

    # 3. Ép kiểu Severity sang số nguyên (Integer) trước khi so sánh cho chắc chắn
    df = df.withColumn("Severity", col("Severity").cast("integer"))

    return df.withColumn("label", when(col("Severity") >= 3, 1).otherwise(0))