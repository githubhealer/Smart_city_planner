CREATE PYTHON3 SET SCRIPT "RF_RAIN_PREDICT" (
    "DATA_ORDER" DECIMAL(1,0),
    "DATASET" VARCHAR(10) UTF8,
    "OBS_DATE" DATE,
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
    "OBS_DATE_OUT" DATE,
    "ACTUAL_RAIN" DECIMAL(18,0),
    "RAIN_PROBABILITY" DOUBLE,
    "PREDICTED_RAIN" DECIMAL(18,0)
)
AS

from sklearn.ensemble import RandomForestClassifier
import numpy as np

def run(ctx):

    train_X = []
    train_y = []
    test_rows = []

    while ctx.next():

        X = [
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
        ]

        if ctx.DATA_ORDER == 1:

            train_X.append(X)
            train_y.append(int(ctx.RAIN_T_PLUS_2))

        else:

            test_rows.append((
                ctx.OBS_DATE,
                int(ctx.RAIN_T_PLUS_2),
                X
            ))

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        np.array(train_X, dtype=float),
        np.array(train_y, dtype=int)
    )

    for obs_date, actual, X in test_rows:

        probability = float(
            model.predict_proba(
                np.array([X], dtype=float)
            )[0][1]
        )

        predicted = 1 if probability >= 0.30 else 0

        ctx.emit(
            obs_date,
            actual,
            probability,
            predicted
        )
/
