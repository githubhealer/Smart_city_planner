CREATE PYTHON3 SET SCRIPT "TRAIN_RAIN_MODEL" (
    "TEMP_AVG" DOUBLE,
    "TEMP_MIN" DOUBLE,
    "TEMP_MAX" DOUBLE,
    "HUMIDITY_AVG" DOUBLE,
    "DEW_POINT_AVG" DOUBLE,
    "APPARENT_TEMP_AVG" DOUBLE,
    "PRECIP_TOTAL" DOUBLE,
    "RAIN_TOTAL" DOUBLE,
    "PRESSURE_AVG" DOUBLE,
    "CLOUD_COVER_AVG" DOUBLE,
    "WIND_SPEED_AVG" DOUBLE,
    "WIND_SPEED_MAX" DOUBLE,
    "WIND_GUST_MAX" DOUBLE,
    "WIND_DIR_SIN" DOUBLE,
    "WIND_DIR_COS" DOUBLE,
    "SOLAR_RADIATION_AVG" DOUBLE,
    "RAIN_T_PLUS_2" DECIMAL(1,0)
)
EMITS (
    "PARAM_NAME" VARCHAR(100) UTF8,
    "PARAM_VALUE" DOUBLE
)
AS

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

def run(ctx):

    X = []
    y = []

    while ctx.next():

        X.append([
            ctx.TEMP_AVG,
            ctx.TEMP_MIN,
            ctx.TEMP_MAX,
            ctx.HUMIDITY_AVG,
            ctx.DEW_POINT_AVG,
            ctx.APPARENT_TEMP_AVG,
            ctx.PRECIP_TOTAL,
            ctx.RAIN_TOTAL,
            ctx.PRESSURE_AVG,
            ctx.CLOUD_COVER_AVG,
            ctx.WIND_SPEED_AVG,
            ctx.WIND_SPEED_MAX,
            ctx.WIND_GUST_MAX,
            ctx.WIND_DIR_SIN,
            ctx.WIND_DIR_COS,
            ctx.SOLAR_RADIATION_AVG
        ])

        y.append(int(ctx.RAIN_T_PLUS_2))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_scaled, y)

    # Model parameters
    ctx.emit(
        "INTERCEPT",
        float(model.intercept_[0])
    )

    for i, coefficient in enumerate(model.coef_[0]):

        ctx.emit(
            "COEF_" + str(i),
            float(coefficient)
        )

    # Scaler parameters
    for i in range(len(scaler.mean_)):

        ctx.emit(
            "MEAN_" + str(i),
            float(scaler.mean_[i])
        )

        ctx.emit(
            "SCALE_" + str(i),
            float(scaler.scale_[i])
        )
/
