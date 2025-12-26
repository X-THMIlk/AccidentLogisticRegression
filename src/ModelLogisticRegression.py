from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression
import pickle
import os

# 1) Hàm chuẩn bị feature
def prepare_features(train_df, test_df, feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)
    
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(train_df)
    train_df = scaler_model.transform(train_df)
    test_df = scaler_model.transform(test_df)
    
    return train_df, test_df, scaler_model

# 2) Hàm train model

def train_model(df, feature_cols):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    # In số lượng mẫu mỗi tập
    train_count = train_df.count()
    test_count = test_df.count()
    total_count = df.count()
    print(f"Total samples: {total_count}")
    print(f"Train samples: {train_count} ({train_count / total_count * 100:.1f}%)")
    print(f"Test samples: {test_count} ({test_count / total_count * 100:.1f}%)")
    
    train_df, test_df, scaler_model = prepare_features(train_df, test_df, feature_cols)
    # Logistic Regression
    lr = LogisticRegression(
        featuresCol="features", 
        labelCol="label", 
        maxIter=50, 
        regParam=0.01, 
        elasticNetParam=0.0
    )
    # Train model
    model = lr.fit(train_df)
    #Dự đoán trên tập test
    predictions = model.transform(test_df)
    return model, predictions, scaler_model


# 3) Hàm lưu model .pkl

def save_model_pkl(model, scaler_model, feature_cols, pkl_path="../models/logistic_model.pkl"):
    os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
    
    coefficients = model.coefficients.toArray().tolist()
    intercept = float(model.intercept)
    
    scaler_mean = scaler_model.mean.toArray().tolist() if scaler_model.mean is not None else None
    scaler_std = scaler_model.std.toArray().tolist() if hasattr(scaler_model, 'std') and scaler_model.std is not None else None
    
    model_data = {
        "coefficients": coefficients,
        "intercept": intercept,
        "feature_cols": feature_cols,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "model_type": "LogisticRegression"
    }
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Lưu model .pkl thành công tại: {os.path.normpath(pkl_path)}")