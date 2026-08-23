CREATE PYTHON3 SET SCRIPT "TRAIN_RF_MODEL" (
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

from sklearn.ensemble import RandomForestClassifier
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

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    ctx.emit(
        "N_ESTIMATORS",
        float(model.n_estimators)
    )

    ctx.emit(
        "MAX_DEPTH",
        float(model.max_depth)
    )

    for i, importance in enumerate(model.feature_importances_):

        ctx.emit(
            "FEATURE_IMPORTANCE_" + str(i),
            float(importance)
        )
/
