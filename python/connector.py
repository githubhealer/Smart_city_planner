import os
import streamlit as st
import boto3
import pyexasol
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Milan Smart City",
    layout="centered"
)

# ============================================================
# AWS BEDROCK
# ============================================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION")
)

# ============================================================
# EXASOL
# ============================================================

def get_exasol_connection():
    return pyexasol.connect(
        dsn=os.getenv("EXASOL_DSN"),
        user=os.getenv("EXASOL_USER"),
        password=os.getenv("EXASOL_PASSWORD"),
        schema=os.getenv("EXASOL_SCHEMA"),
        websocket_sslopt={"cert_reqs": 0}
    )

# ============================================================
# RAIN PREDICTION
# ============================================================

def get_rain_prediction(prediction_date):
    conn = get_exasol_connection()

    query = f"""
    SELECT
        OBS_DATE_OUT,
        RAIN_PROBABILITY,
        PREDICTED_RAIN
    FROM MILAN_WEATHER.MILAN_RF_PREDICTIONS
    WHERE OBS_DATE_OUT = DATE '{prediction_date}'
    """

    result = conn.execute(query)
    row = result.fetchone()
    conn.close()

    if row is None:
        return {
            "status": "error",
            "message": f"No prediction available for {prediction_date}."
        }

    return {
        "status": "success",
        "prediction_date": str(row[0]),
        "rain_probability": float(row[1]),
        "predicted_rain": int(row[2])
    }

# ============================================================
# NOVA TOOL
# ============================================================

tool_config = {
    "tools": [{
        "toolSpec": {
            "name": "get_rain_prediction",
            "description": (
                "Gets the Random Forest rain prediction "
                "from the Exasol Smart City weather model "
                "for a specific date provided by the user."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "prediction_date": {
                            "type": "string",
                            "description": (
                                "Date for prediction in "
                                "YYYY-MM-DD format."
                            )
                        }
                    },
                    "required": ["prediction_date"]
                }
            }
        }
    }],
    "toolChoice": {
        "tool": {
            "name": "get_rain_prediction"
        }
    }
}

# ============================================================
# FINAL NOVA TOOL CONFIG
# ============================================================

tool_config_final = {
    "tools": tool_config["tools"]
}

# ============================================================
# NOVA PROCESSING
# ============================================================

def process_request(user_input):
    messages = [{
        "role": "user",
        "content": [{
            "text": f"""
You are a Smart City weather assistant.

The user will provide a date and information about
their plans.

Extract the requested date and convert it to
YYYY-MM-DD format.

You MUST use the get_rain_prediction tool for that date.

After receiving the prediction, provide ONLY the
final answer to the user.

Do not describe your reasoning.
Do not mention the tool.
Do not explain how you extracted the date.
Do not repeat the user's question.

Give concise, practical advice based on:
- rain probability
- whether rain is predicted
- the user's schedule
- walking or outdoor activities

Do not invent weather information.

User request:

{user_input}
"""
        }]
    }]

    # First Nova call
    response = bedrock.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=messages,
        toolConfig=tool_config,
        inferenceConfig={
            "temperature": 0,
            "maxTokens": 500
        }
    )

    output_message = response["output"]["message"]
    messages.append(output_message)

    # Tool call
    for content in output_message["content"]:
        if "toolUse" not in content:
            continue

        tool_use = content["toolUse"]

        result = get_rain_prediction(
            tool_use["input"]["prediction_date"]
        )

        messages.append({
            "role": "user",
            "content": [{
                "toolResult": {
                    "toolUseId": tool_use["toolUseId"],
                    "content": [{"json": result}]
                }
            }]
        })

    # Second Nova call
    final_response = bedrock.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=messages,
        toolConfig=tool_config_final,
        inferenceConfig={
            "temperature": 0.2,
            "maxTokens": 500
        }
    )

    # Get final text
    final_text = ""

    for content in final_response["output"]["message"]["content"]:
        if "text" in content:
            final_text += content["text"]

    return final_text.strip()

# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>

    /* Entire application */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    /* Main content */
    .main .block-container {
        max-width: 900px;
        padding-top: 60px;
        padding-bottom: 50px;
    }

    /* Title */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 18px;
        margin-bottom: 40px;
    }

    /* Text area label */
    label {
        color: #ffffff !important;
        font-size: 16px !important;
    }

    /* Text area */
    textarea {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
    }

    textarea::placeholder {
        color: #777777 !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background-color: #ffffff;
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 600;
        margin-top: 10px;
    }

    .stButton > button:hover {
        background-color: #dddddd;
        color: #000000;
    }

    /* Result card */
    .result-card {
        background-color: #111111;
        color: #ffffff !important;
        border: 1px solid #333333;
        padding: 24px;
        border-radius: 12px;
        margin-top: 30px;
        font-size: 18px;
        line-height: 1.6;
    }

    .result-card p {
        color: #ffffff !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #ffffff !important;
    }

    /* Warning / error text */
    .stAlert {
        background-color: #111111 !important;
        color: #ffffff !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Milan Smart City</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Rain & Travel Assistant</div>',
    unsafe_allow_html=True
)

# ============================================================
# USER INPUT
# ============================================================

user_input = st.text_area(
    "Tell me about your plans",
    placeholder=(
        "I'm going to college on 2023-06-16. "
        "I leave home at 6 AM and return at 7 PM. "
        "I have to walk for 20 minutes. "
        "Tell me if I should prepare for rain."
    ),
    height=140
)

# ============================================================
# CHECK BUTTON
# ============================================================

if st.button("Check My Day", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter your plans.")

    else:
        with st.spinner("Checking the weather prediction..."):
            try:
                answer = process_request(user_input)

                st.markdown(
                    f"""
                    <div class="result-card">
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(
                    f"Unable to process the request: {e}"
                )
