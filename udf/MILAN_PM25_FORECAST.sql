CREATE PYTHON3 SET SCRIPT "STARTER_KIT"."MILAN_PM25_FORECAST" (
    "DATA_ORDER" DECIMAL(1,0),
    "DATETIME" TIMESTAMP,
    "PM2_5" DOUBLE,
    "PM10" DOUBLE,
    "NITROGEN_DIOXIDE" DOUBLE,
    "SULPHUR_DIOXIDE" DOUBLE,
    "OZONE" DOUBLE,
    "HOUR" DECIMAL(2,0),
    "DAY_OF_WEEK" DECIMAL(1,0),
    "MONTH" DECIMAL(2,0),
    "TARGET_PM2_5" DOUBLE
)
EMITS (
    "DATETIME_OUT" TIMESTAMP,
    "CURRENT_PM2_5" DOUBLE,
    "ACTUAL_PM2_5_NEXT_HOUR" DOUBLE,
    "PREDICTED_PM2_5_NEXT_HOUR" DOUBLE,
    "PREDICTION_ERROR" DOUBLE
)
AS

from xgboost import XGBRegressor
import numpy as np


def run(ctx):

    train_X = []
    train_y = []
    test_rows = []

    while ctx.next():

        X = [
            ctx.PM2_5,
            ctx.PM10,
            ctx.NITROGEN_DIOXIDE,
            ctx.SULPHUR_DIOXIDE,
            ctx.OZONE,
            ctx.HOUR,
            ctx.DAY_OF_WEEK,
            ctx.MONTH
        ]

        if ctx.DATA_ORDER == 1:

            train_X.append(X)
            train_y.append(ctx.TARGET_PM2_5)

        else:

            test_rows.append(
                (
                    ctx.DATETIME,
                    ctx.PM2_5,
                    ctx.TARGET_PM2_5,
                    X
                )
            )

    X_train = np.array(train_X, dtype=float)
    y_train = np.array(train_y, dtype=float)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    for obs_datetime, current_pm25, actual, X in test_rows:

        prediction = float(
            model.predict(
                np.array([X], dtype=float)
            )[0]
        )

        error = float(actual - prediction)

        ctx.emit(
            obs_datetime,
            current_pm25,
            actual,
            prediction,
            error
        )

/

