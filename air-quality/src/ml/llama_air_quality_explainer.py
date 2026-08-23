import requests
import pandas as pd

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/home/asus/exasol-air-quality")

RESULT_DIR = BASE_DIR / "results/milan"

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "llama3.2:3b"


# ============================================================
# LOAD XGBOOST PREDICTIONS
# ============================================================

prediction_file = (
    RESULT_DIR /
    "next_hour_pm25_predictions.csv"
)

df = pd.read_csv(prediction_file)

print("Prediction rows loaded:", len(df))


# ============================================================
# PROJECT AIR-QUALITY SCORE
# ============================================================

def calculate_air_quality_score(pm25):
    """
    Project-specific score.

    100 = lower predicted PM2.5
    0   = very high predicted PM2.5

    This is NOT an official regulatory AQI.
    """

    score = 100 - pm25

    return max(0, min(100, score))


df["air_quality_score"] = (
    df["predicted_pm2_5_next_hour"]
    .apply(calculate_air_quality_score)
)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def calculate_risk(score):

    if score < 20:
        return "CRITICAL"

    elif score < 40:
        return "HIGH"

    elif score < 60:
        return "MODERATE"

    else:
        return "LOW"


df["risk_level"] = (
    df["air_quality_score"]
    .apply(calculate_risk)
)


# ============================================================
# SELECT IMPORTANT EVENTS
# ============================================================

events = (
    df[
        df["risk_level"].isin(
            ["CRITICAL", "HIGH"]
        )
    ]
    .sort_values(
        "air_quality_score"
    )
    .head(10)
)

print(
    "High-risk events selected:",
    len(events)
)


# ============================================================
# LLAMA FUNCTION
# ============================================================

def ask_llama(prompt):

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"]


# ============================================================
# GENERATE EXPLANATIONS
# ============================================================

results = []


for _, row in events.iterrows():

    prompt = f"""
You are an AI decision-support assistant for an
urban air-quality monitoring system.

The numerical prediction was produced by an XGBoost
machine-learning model.

Your job is ONLY to explain the supplied results and
suggest reasonable actions for city authorities.

IMPORTANT RULES:

1. Do not change any supplied numerical values.
2. Do not invent measurements.
3. Do not calculate a different air-quality score.
4. Do not call this project-specific score an official AQI.
5. Do not claim that one pollutant definitely caused another.
6. Treat pollutant values as indicators/contributors,
   not proof of causation.
7. Do not invent traffic or weather conditions.
8. Keep recommendations practical and suitable for
   city authorities.
9. Do not provide medical diagnosis.
10. Use the risk level already calculated by our system.

--------------------------------------------------

EVENT INFORMATION

Timestamp:
{row["datetime"]}

Current PM2.5:
{row["pm2_5"]:.2f} µg/m³

Predicted next-hour PM2.5:
{row["predicted_pm2_5_next_hour"]:.2f} µg/m³

PM10:
{row["pm10"]:.2f} µg/m³

Nitrogen dioxide:
{row["nitrogen_dioxide"]:.2f} µg/m³

Sulphur dioxide:
{row["sulphur_dioxide"]:.2f} µg/m³

Ozone:
{row["ozone"]:.2f} µg/m³

Project Air-Quality Score:
{row["air_quality_score"]:.2f}/100

Risk Level:
{row["risk_level"]}

--------------------------------------------------

Respond using exactly these sections:

SITUATION

Explain what the prediction means in 2-3 sentences.

CONTRIBUTING POLLUTANTS

Identify the pollutants that are elevated or
particularly relevant in this event.

POTENTIAL IMPACT

Explain the potential significance of the
predicted pollution episode.

RECOMMENDED ACTIONS

Give 3-5 practical actions city authorities
could consider.

MONITORING

Explain what should be monitored during the
next few hours.

Keep the answer concise and suitable for an
AI hackathon dashboard.
"""

    try:

        explanation = ask_llama(prompt)

    except Exception as e:

        explanation = (
            "Llama explanation failed: "
            + str(e)
        )


    results.append({
        "datetime": row["datetime"],
        "current_pm2_5": row["pm2_5"],
        "predicted_pm2_5_next_hour":
            row["predicted_pm2_5_next_hour"],
        "pm10": row["pm10"],
        "nitrogen_dioxide":
            row["nitrogen_dioxide"],
        "sulphur_dioxide":
            row["sulphur_dioxide"],
        "ozone":
            row["ozone"],
        "air_quality_score":
            row["air_quality_score"],
        "risk_level":
            row["risk_level"],
        "llama_explanation":
            explanation
    })


# ============================================================
# SAVE
# ============================================================

output = pd.DataFrame(results)

output_file = (
    RESULT_DIR /
    "llama_air_quality_explanations.csv"
)

output.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n========================================")
print("LLAMA AIR QUALITY EXPLANATION")
print("========================================")

print(
    "Explanations generated:",
    len(output)
)

print("\nSaved:")
print(output_file)


if len(output) > 0:

    first = output.iloc[0]

    print("\n----------------------------------------")
    print("EXAMPLE EVENT")
    print("----------------------------------------")

    print(
        "Timestamp:",
        first["datetime"]
    )

    print(
        f"Predicted PM2.5: "
        f"{first['predicted_pm2_5_next_hour']:.2f}"
        " µg/m³"
    )

    print(
        f"Air Quality Score: "
        f"{first['air_quality_score']:.2f}/100"
    )

    print(
        "Risk:",
        first["risk_level"]
    )

    print("\nLLAMA EXPLANATION:\n")

    print(
        first["llama_explanation"]
    )


print("\n========================================")
print("LLM EXPLANATION COMPLETE")
print("========================================")
