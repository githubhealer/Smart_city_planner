import pyexasol
import ssl
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/home/asus/exasol-air-quality")

MODEL_DIR = BASE_DIR / "models/milan"
RESULT_DIR = BASE_DIR / "results/milan"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONNECT TO EXASOL
# ============================================================

password = (
    Path.home()
    / ".exasol-starter-kit/credentials/nano_sys_password"
).read_text().strip()

conn = pyexasol.connect(
    dsn="127.0.0.1:8563",
    user="sys",
    password=password,
    encryption=True,
    protocol_version=3,
    websocket_sslopt={
        "cert_reqs": ssl.CERT_NONE
    },
)

print("Connected to Exasol")


# ============================================================
# LOAD AIR QUALITY DATA
# ============================================================

query = """
SELECT
    "datetime",
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour",
    "day_of_week",
    "month"
FROM STARTER_KIT.MILAN_AIR_QUALITY_CLEAN
ORDER BY "datetime"
"""

rows = conn.execute(query).fetchall()

columns = [
    "datetime",
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour",
    "day_of_week",
    "month",
]

df = pd.DataFrame(rows, columns=columns)

conn.close()

print("Rows loaded from Exasol:", len(df))


# ============================================================
# PREPARE DATA
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"])

numeric_columns = [
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour",
    "day_of_week",
    "month",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().copy()

df = df.sort_values("datetime").reset_index(drop=True)


# ============================================================
# CREATE NEXT-HOUR TARGET
# ============================================================

# Target = PM2.5 concentration one hour into the future

df["target_pm2_5_next_hour"] = df["pm2_5"].shift(-1)

df = df.dropna().reset_index(drop=True)


# ============================================================
# FEATURES
# ============================================================

features = [
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour",
    "day_of_week",
    "month",
]

X = df[features]
y = df["target_pm2_5_next_hour"]


print("Rows available for modelling:", len(df))


# ============================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

# IMPORTANT:
# We do NOT randomly shuffle the data.
# Earlier observations = training
# Later observations = testing

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

print(
    "Training period:",
    df["datetime"].iloc[0],
    "to",
    df["datetime"].iloc[split_index - 1],
)

print(
    "Testing period :",
    df["datetime"].iloc[split_index],
    "to",
    df["datetime"].iloc[-1],
)


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

within_5 = (
    np.abs(y_test.values - y_pred) <= 5
).mean() * 100

within_10 = (
    np.abs(y_test.values - y_pred) <= 10
).mean() * 100


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("MILAN NEXT-HOUR PM2.5 FORECAST")
print("========================================")

print(f"MAE                  : {mae:.3f} µg/m³")
print(f"RMSE                 : {rmse:.3f} µg/m³")
print(f"R²                   : {r2:.3f}")
print(f"Predictions ±5 µg/m³ : {within_5:.2f}%")
print(f"Predictions ±10 µg/m³: {within_10:.2f}%")

print("========================================")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "xgboost_next_hour_pm25.pkl"
)

joblib.dump(
    {
        "model": model,
        "features": features,
        "target": "target_pm2_5_next_hour",
    },
    model_path,
)

print("\nModel saved:")
print(model_path)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = df.iloc[split_index:].copy()

prediction_df["actual_pm2_5_next_hour"] = y_test.values

prediction_df["predicted_pm2_5_next_hour"] = y_pred

prediction_df["prediction_error"] = (
    prediction_df["actual_pm2_5_next_hour"]
    - prediction_df["predicted_pm2_5_next_hour"]
)

prediction_path = (
    RESULT_DIR /
    "next_hour_pm25_predictions.csv"
)

prediction_df[
    [
        "datetime",
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "actual_pm2_5_next_hour",
        "predicted_pm2_5_next_hour",
        "prediction_error",
    ]
].to_csv(
    prediction_path,
    index=False,
)

print("\nPredictions saved:")
print(prediction_path)


# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "metric": [
        "MAE",
        "RMSE",
        "R2",
        "Predictions_within_5_ug_m3_percent",
        "Predictions_within_10_ug_m3_percent",
    ],
    "value": [
        mae,
        rmse,
        r2,
        within_5,
        within_10,
    ],
})

metrics_path = (
    RESULT_DIR /
    "forecast_evaluation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)

print("\nEvaluation metrics saved:")
print(metrics_path)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
})

importance_df["importance_percent"] = (
    importance_df["importance"]
    / importance_df["importance"].sum()
    * 100
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

importance_path = (
    RESULT_DIR /
    "forecast_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False,
)

print("\nFeature importance saved:")
print(importance_path)

print("\nTOP FEATURES")
print(
    importance_df.to_string(index=False)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("FORECAST MODEL TRAINING COMPLETE")
print("========================================")
