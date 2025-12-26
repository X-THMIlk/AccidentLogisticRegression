import pandas as pd
import numpy as np
import os
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def evaluate_model(pred):
    """
    Evaluation for US Traffic Accident Severity Prediction
    Positive class (1): Severe accident (Severity 3-4)
    Negative class (0): Minor accident (Severity 1-2)
    """

    # ======================
    # BASIC METRICS
    # ======================
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label",
        metricName="areaUnderROC"
    )
    auc_roc = evaluator_auc.evaluate(pred)

    evaluator = MulticlassClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(pred, {evaluator.metricName: "accuracy"})
    weighted_precision = evaluator.evaluate(pred, {evaluator.metricName: "weightedPrecision"})
    weighted_recall = evaluator.evaluate(pred, {evaluator.metricName: "weightedRecall"})
    weighted_f1 = evaluator.evaluate(pred, {evaluator.metricName: "f1"})

    # ======================
    # CONFUSION MATRIX
    # ======================
    cm = pred.groupBy("label", "prediction").count().toPandas()

    TP = cm[(cm.label == 1) & (cm.prediction == 1)]["count"].sum() if not cm.empty else 0
    TN = cm[(cm.label == 0) & (cm.prediction == 0)]["count"].sum() if not cm.empty else 0
    FP = cm[(cm.label == 0) & (cm.prediction == 1)]["count"].sum() if not cm.empty else 0
    FN = cm[(cm.label == 1) & (cm.prediction == 0)]["count"].sum() if not cm.empty else 0

    # ======================
    # RATE METRICS (TRAFFIC SAFETY FOCUS)
    # ======================
    sensitivity = TP / (TP + FN + 1e-9)      # Recall – detect severe accidents
    specificity = TN / (TN + FP + 1e-9)      # Detect non-severe accidents
    fpr = FP / (FP + TN + 1e-9)
    fnr = FN / (TP + FN + 1e-9)

    precision = TP / (TP + FP + 1e-9)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)

    # ======================
    # PROBABILITY METRICS
    # ======================
    # Chuyển đổi cột probability (vector) sang mảng numpy để tính toán
    prob_list = pred.select("probability").toPandas()["probability"]
    # Lấy xác suất của lớp Positive (index 1)
    prob = np.array([float(p[1]) for p in prob_list])
    y_true = pred.select("label").toPandas()["label"].values

    brier_score = np.mean((prob - y_true) ** 2)
    log_loss = -np.mean(
        y_true * np.log(prob + 1e-15) +
        (1 - y_true) * np.log(1 - prob + 1e-15)
    )

    # Matthews Correlation Coefficient
    mcc = (TP * TN - FP * FN) / np.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-9
    )

    youdens_j = sensitivity - fpr

    # ======================
    # METRICS SUMMARY
    # ======================
    metrics = {
        "Accuracy": accuracy,
        "AUC-ROC": auc_roc,

        "Precision (Severe Accident)": precision,
        "Recall / Sensitivity (Severe Accident Detection)": sensitivity,
        "Specificity (Non-Severe Detection)": specificity,
        "F1-Score": f1,

        "False Negative Rate (Missed Severe Accidents)": fnr,
        "False Positive Rate": fpr,

        "MCC": mcc,
        "Youdens J Statistic": youdens_j,
        "Brier Score": brier_score,
        "Log Loss": log_loss,

        "TP (Severe Accidents Detected)": int(TP),
        "TN (Minor Accidents Detected)": int(TN),
        "FP": int(FP),
        "FN (Missed Severe Accidents)": int(FN)
    }
    # In chi tiết đánh giá mô hình US Accidents
    print("\n" + "=" * 60)
    print("US ACCIDENTS MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n🔹 BASIC METRICS:")
    print(f"  Accuracy:        {metrics['Accuracy']:.4f}")
    print(f"  AUC-ROC:         {metrics['AUC-ROC']:.4f}")

    print(f"\n🔹 SEVERE ACCIDENT DETECTION METRICS:")
    print(f"  Precision (Severe Accident):       {metrics['Precision (Severe Accident)']:.4f}")
    print(f"  Recall / Sensitivity (Severe Accident Detection): {metrics['Recall / Sensitivity (Severe Accident Detection)']:.4f}")
    print(f"  Specificity (Non-Severe Detection): {metrics['Specificity (Non-Severe Detection)']:.4f}")
    print(f"  F1-Score:                          {metrics['F1-Score']:.4f}")

    print(f"\n🔹 ADVANCED METRICS:")
    print(f"  MCC:             {metrics['MCC']:.4f}")
    print(f"  Youden's J:      {metrics['Youdens J Statistic']:.4f}")
    print(f"  Brier Score:     {metrics['Brier Score']:.4f}")
    print(f"  Log Loss:        {metrics['Log Loss']:.4f}")

    print(f"\n🔹 CONFUSION MATRIX:")
    print(f"  TP (Severe Accidents Detected): {int(metrics['TP (Severe Accidents Detected)'])}")
    print(f"  TN (Minor Accidents Detected):  {int(metrics['TN (Minor Accidents Detected)'])}")
    print(f"  FP: {int(metrics['FP'])}")
    print(f"  FN (Missed Severe Accidents):   {int(metrics['FN (Missed Severe Accidents)'])}")
    print(f"  Total Accidents: {int(metrics['TP (Severe Accidents Detected)'] + metrics['TN (Minor Accidents Detected)'] + metrics['FP'] + metrics['FN (Missed Severe Accidents)'])}")
    print("=" * 60 + "\n")


    return metrics, cm

def save_evaluation_to_file(metrics, cm):
    """
    Save evaluation results for US Traffic Accident Severity Prediction
    """

    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    RESULT_DIR = os.path.join(BASE_DIR, "results")

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ======================
    # Save metrics
    # ======================
    metrics_df = pd.DataFrame([
        {"Metric": k, "Value": round(v, 6) if isinstance(v, float) else v}
        for k, v in metrics.items()
    ])

    metrics_df.to_csv(
        os.path.join(RESULT_DIR, "us_accidents_model_evaluation.csv"),
        index=False
    )
    # ======================
    # Save confusion matrix
    # ======================
    cm_sorted = cm.sort_values(by=["label", "prediction"])
    cm_sorted.to_csv(
        os.path.join(RESULT_DIR, "us_accidents_confusion_matrix.csv"),
        index=False
    )

    print("✅ Evaluation results saved successfully:")
    print(f"   - {RESULT_DIR}/us_accidents_model_evaluation.csv")
    print(f"   - {RESULT_DIR}/us_accidents_confusion_matrix.csv")