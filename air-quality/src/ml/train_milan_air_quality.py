import pyexasol
import ssl
import joblib

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/home/asus/exasol-air-quality")

MODEL_DIR = BASE_DIR / "models/milan"
RESULT_DIR = BASE_DIR / "results/milan"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_PATH = MODEL_DIR / "xgboost_air_quality.pkl"

PREDICTIONS_PATH = RESULT_DIR / "predictions.csv"

METRICS_PATH = RESULT_DIR / "evaluation_metrics.csv"

FEATURE_IMPORTANCE_PATH = RESULT_DIR / "feature_importance.csv"


# ============================================================
# EXASOL CONNECTION
# ============================================================

password_file = (
    Path.home()
    / ".exasol-starter-kit/credentials/nano_sys_password"
)

password = password_file.read_text().strip()


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
# LOAD DATA
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
# CONVERT NUMERIC COLUMNS
# ============================================================

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

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# CREATE PROJECT-SPECIFIC AIR QUALITY TARGET
# ============================================================

# Higher pollutant concentration = worse air quality.
#
# We normalize each pollutant using its 95th percentile.
# This prevents extreme values from completely dominating
# the score.
#
# The pollutant components are combined into a pollution
# severity score from approximately 0-100.
#
# Then:
#
# AI Air Quality Score = 100 - Pollution Severity


pollutants = [
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]


for column in pollutants:

    reference_value = df[column].quantile(0.95)

    df[f"{column}_normalized"] = (
        df[column] / reference_value
    ).clip(0, 1)


# ============================================================
# POLLUTION SEVERITY
# ============================================================

# Weights represent the contribution of each pollutant
# to the project-specific pollution score.

weights = {
    "pm2_5_normalized": 0.30,
    "pm10_normalized": 0.25,
    "nitrogen_dioxide_normalized": 0.20,
    "ozone_normalized": 0.15,
    "sulphur_dioxide_normalized": 0.10,
}


df["pollution_severity"] = 0

for feature, weight in weights.items():

    df["pollution_severity"] += (
        df[feature] * weight
    )


df["pollution_severity"] = (
    df["pollution_severity"] * 100
).clip(0, 100)


# ============================================================
# TARGET
# ============================================================

df["ai_air_quality_score"] = (
    100 - df["pollution_severity"]
).clip(0, 100)


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

y = df["ai_air_quality_score"]


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

valid_rows = y.notna()

X = X.loc[valid_rows].copy()

y = y.loc[valid_rows].copy()

df = df.loc[valid_rows].copy()


print("Rows available for modelling:", len(df))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


print("Training rows:", len(X_train))

print("Testing rows:", len(X_test))


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(X_train)

X_test_imputed = imputer.transform(X_test)


# ============================================================
# XGBOOST
# ============================================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=2,
)


print("\nTraining XGBoost...")

model.fit(
    X_train_imputed,
    y_train,
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_imputed
)

y_pred = np.clip(
    y_pred,
    0,
    100
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

within_5 = (
    np.abs(
        y_test.values - y_pred
    ) <= 5
).mean() * 100

within_10 = (
    np.abs(
        y_test.values - y_pred
    ) <= 10
).mean() * 100


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("MILAN XGBOOST MODEL RESULTS")
print("========================================")

print(f"MAE                  : {mae:.3f}")

print(f"RMSE                 : {rmse:.3f}")

print(f"R²                   : {r2:.3f}")

print(
    f"Predictions ±5 pts  : {within_5:.2f}%"
)

print(
    f"Predictions ±10 pts : {within_10:.2f}%"
)

print("========================================")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "features": features,
    },
    MODEL_PATH,
)


print("\nModel saved:")

print(MODEL_PATH)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = df.loc[
    X_test.index,
    [
        "datetime",
        "pm10",
        "pm2_5",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "pollution_severity",
        "ai_air_quality_score",
    ],
].copy()


prediction_df["predicted_ai_air_quality_score"] = y_pred

prediction_df["prediction_error"] = (
    prediction_df["ai_air_quality_score"]
    - prediction_df[
        "predicted_ai_air_quality_score"
    ]
)


prediction_df["health_score"] = (
    prediction_df[
        "predicted_ai_air_quality_score"
    ]
)


prediction_df["pollution_severity_predicted"] = (
    100
    - prediction_df["health_score"]
)


prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


print("\nPredictions saved:")

print(PREDICTIONS_PATH)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    {
        "metric": [
            "MAE",
            "RMSE",
            "R2",
            "Predictions_within_5_points_percent",
            "Predictions_within_10_points_percent",
        ],
        "value": [
            mae,
            rmse,
            r2,
            within_5,
            within_10,
        ],
    }
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


print("\nEvaluation metrics saved:")

print(METRICS_PATH)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature": features,
        "importance": model.feature_importances_,
    }
)

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


importance_df["importance_percent"] = (
    importance_df["importance"]
    / importance_df["importance"].sum()
) * 100


importance_df.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False,
)


print("\nFeature importance saved:")

print(FEATURE_IMPORTANCE_PATH)


print("\n========================================")
print("MILAN MODEL TRAINING COMPLETE")
print("========================================")
