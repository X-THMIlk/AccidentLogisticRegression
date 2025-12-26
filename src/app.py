from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import os, subprocess, pickle, numpy as np, pandas as pd

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Tạo thư mục nếu chưa có
for d in [DATA_DIR, IMAGE_DIR, RESULT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(SRC_DIR, "views"))


# ================= ROUTES TĨNH =================
@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULT_DIR, filename)


# ================= ROUTES CHÍNH =================

@app.route("/")
def home():
    return render_template("upload.html")

#Upload
@app.route("/upload", methods=["POST"])
def upload():
    try:
        f = request.files["file"]
        if f:
            # Lưu file upload
            file_path = os.path.join(DATA_DIR, "data.csv")
            f.save(file_path)

            # Chạy MainModel.py để train lại (Lưu ý: Chạy ngầm)
            subprocess.run(["python", os.path.join(SRC_DIR, "MainModel.py")], check=True)
            return redirect("/training")
    except Exception as e:
        print(f"Lỗi upload: {e}")
    return redirect("/")


@app.route("/training")
def training():
    return render_template("training.html")


#Evaluation
@app.route("/evaluation")
def evaluation():
    # --- 1. ĐỌC METRICS ---
    metrics = {}
    eval_path = os.path.join(RESULT_DIR, "us_accidents_model_evaluation.csv")

    if os.path.exists(eval_path):
        try:
            df = pd.read_csv(eval_path)

            # [QUAN TRỌNG] Chuyển đổi 2 cột 'Metric' và 'Value' thành Dictionary {Key: Value}
            # File CSV có dạng:
            # Metric        | Value
            # Accuracy      | 0.8063
            # AUC-ROC       | 0.5872
            # ...

            # Lệnh này sẽ tạo ra dict: {'Accuracy': 0.8063, 'AUC-ROC': 0.5872, ...}
            raw_metrics = dict(zip(df["Metric"], df["Value"]))

            # Map lại các key dài dòng trong CSV sang key ngắn gọn cho HTML
            metrics = {
                "Accuracy": raw_metrics.get("Accuracy", 0),

                # Sửa lỗi: Map đúng tên key dài trong CSV sang key ngắn "Precision"
                "Precision": raw_metrics.get("Precision (Severe Accident)", 0),

                # Sửa lỗi: Map đúng tên key dài trong CSV sang key ngắn "Recall"
                "Recall": raw_metrics.get("Recall / Sensitivity (Severe Accident Detection)", 0),

                "F1-Score": raw_metrics.get("F1-Score", 0),
                "AUC-ROC": raw_metrics.get("AUC-ROC", 0)
            }

        except Exception as e:
            print(f"Lỗi đọc metrics: {e}")
            # Giá trị mặc định nếu lỗi
            metrics = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0}
    else:
        print(f"⚠️ Không tìm thấy file metrics tại: {eval_path}")
        metrics = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0}

    # --- 2. ĐỌC CONFUSION MATRIX ---
    cm = [[0, 0], [0, 0]]
    cm_path = os.path.join(RESULT_DIR, "us_accidents_confusion_matrix.csv")

    if os.path.exists(cm_path):
        try:
            df_cm = pd.read_csv(cm_path)
            # Confusion matrix file của bạn lưu 3 cột: label, prediction, count.
            # Cần pivot hoặc reshape lại thành ma trận 2x2.
            # Tuy nhiên, code cũ của bạn dùng df_cm.values.tolist() là chưa chính xác nếu file csv có header.

            # Cách an toàn để tạo ma trận 2x2 từ dữ liệu thống kê metrics nếu có:
            # (Hoặc đọc trực tiếp từ file nếu file đó đã là ma trận)
            # Giả sử file csv confusion matrix lưu dạng bảng đúng chuẩn thì:
            if "count" in df_cm.columns:
                # Nếu file lưu dạng: label, prediction, count (như code save cũ)
                pivot = df_cm.pivot(index="label", columns="prediction", values="count").fillna(0)
                cm = pivot.values.tolist()
            else:
                # Nếu file lưu dạng ma trận thuần
                cm = df_cm.values.tolist()

        except Exception as e:
            print(f"Lỗi đọc CM: {e}")
    else:
        print(f"⚠️ Không tìm thấy file CM tại: {cm_path}")

    return render_template("evaluation.html", metrics=metrics, cm=cm)
#descriptive
@app.route("/descriptive")
def descriptive():
    path = os.path.join(RESULT_DIR, "descriptive_statistics.csv")

    if not os.path.exists(path):
        return render_template(
            "descriptive.html",
            stats=[],
            warning="⚠️ Chưa có thống kê mô tả. Vui lòng upload và huấn luyện dữ liệu trước."
        )

    stats = pd.read_csv(path).to_dict("records")
    return render_template("descriptive.html", stats=stats)

#explain
@app.route("/explain")
def explain():
    model_path = os.path.join(MODEL_DIR, "logistic_model.pkl")
    if not os.path.exists(model_path):
        return "Chưa có model."

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return render_template(
        "explain.html",
        explain=list(zip(model["feature_cols"], model["coefficients"])),
        feature_img="feature_importance.png"
    )

#predict
@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_path = os.path.join(MODEL_DIR, "logistic_model.pkl")

    # Kiểm tra model tồn tại chưa
    if not os.path.exists(model_path):
        return render_template("predict.html", error="⚠️ Chưa có mô hình. Vui lòng train trước.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Danh sách các cột mà model cần (bao gồm cả _idx)
    model_features = model["feature_cols"]

    prob = None
    input_data = {}
    prediction_text = ""
    prediction_label = 0

    if request.method == "POST":
        try:
            # Xử lý input từ form
            input_values = []

            # Bảng map dữ liệu chữ sang số (Mô phỏng lại StringIndexer)
            # Lưu ý: Index này nên khớp với lúc train. Đây là giả định phổ biến.
            mappings = {
                "Weather_Condition": {
                    "Clear": 0, "Cloudy": 1, "Rain": 2, "Snow": 3, "Fog": 4, "Thunderstorm": 5
                },
                "Sunrise_Sunset": {
                    "Day": 0, "Night": 1
                }
            }

            for f in model_features:
                # 1. Xác định tên trường trong Form HTML
                # Model cần "Weather_Condition_idx" nhưng form gửi "Weather_Condition"
                form_field = f.replace("_idx", "")

                # Lấy giá trị từ form
                val = request.form.get(form_field)

                # Lưu lại để hiển thị lại trên form (UX)
                input_data[form_field] = val

                # 2. Chuyển đổi dữ liệu
                if f.endswith("_idx"):
                    # Nếu là cột categorical (chữ), cần map sang số
                    if form_field in mappings and val in mappings[form_field]:
                        val_cleaned = float(mappings[form_field][val])
                    else:
                        val_cleaned = 0.0  # Giá trị mặc định nếu không khớp
                else:
                    # Nếu là số thực bình thường
                    val_cleaned = float(val)

                input_values.append(val_cleaned)

            # 3. Tính toán dự báo
            X = np.array(input_values)

            # Chuẩn hóa (Scaler)
            if model.get("scaler_mean") is not None and model.get("scaler_std") is not None:
                mean = np.array(model["scaler_mean"])
                std = np.array(model["scaler_std"])
                # Tránh chia cho 0
                std = np.where(std == 0, 1, std)
                X = (X - mean) / std

            # Tính toán Logistic Regression: z = w*x + b
            z = np.dot(X, np.array(model["coefficients"])) + model["intercept"]
            prob = 1 / (1 + np.exp(-z))  # Sigmoid

            # Ngưỡng phân loại 0.5
            prediction_label = 1 if prob >= 0.5 else 0
            prediction_text = "NGHIÊM TRỌNG (Severity 3-4)" if prediction_label == 1 else "ÍT NGHIÊM TRỌNG (Severity 1-2)"

        except Exception as e:
            return render_template("predict.html", error=f"Lỗi xử lý dữ liệu: {str(e)}")

    return render_template(
        "predict.html",
        prob=prob,
        input_data=input_data,
        prediction_text=prediction_text,
        prediction_label=prediction_label
    )


if __name__ == "__main__":
    app.run(debug=True)