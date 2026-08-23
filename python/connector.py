import os
import streamlit as st
import boto3
import pyexasol
from dotenv import load_dotenv

load_dotenv()

# Configuration
st.set_page_config(
    page_title="UrbanSense AI",
    layout="centered"
)

# Theme
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# AWS Bedrock
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION")
)

# Exasol connection
def get_exasol_connection():
    return pyexasol.connect(
        dsn=os.getenv("EXASOL_DSN"),
        user=os.getenv("EXASOL_USER"),
        password=os.getenv("EXASOL_PASSWORD"),
        schema=os.getenv("EXASOL_SCHEMA"),
        websocket_sslopt={"cert_reqs": 0}
    )

# Get rain and air quality predictions
def get_city_prediction(prediction_date):
    conn = get_exasol_connection()

    query = f"""
    SELECT
        R.OBS_DATE_OUT,
        R.RAIN_PROBABILITY,
        R.PREDICTED_RAIN,
        A.PREDICTED_AQ_SCORE_AVG,
        A.PREDICTED_AQ_SCORE_MIN,
        A.PREDICTED_AQ_SCORE_MAX,
        A.POLLUTION_SEVERITY_AVG,
        A.POLLUTION_SEVERITY_MAX
    FROM MILAN_WEATHER.MILAN_RF_PREDICTIONS R
    LEFT JOIN MILAN_WEATHER.MILAN_AQ_PREDICTIONS A
        ON R.OBS_DATE_OUT = A.OBS_DATE
    WHERE R.OBS_DATE_OUT = DATE '{prediction_date}'
    """

    result = conn.execute(query)
    row = result.fetchone()
    conn.close()

    if row is None:
        return {
            "status": "error",
            "message": f"No prediction is available for {prediction_date}."
        }

    if row[3] is None:
        return {
            "status": "partial",
            "prediction_date": str(row[0]),
            "rain_probability": float(row[1]),
            "predicted_rain": int(row[2]),
            "air_quality_available": False,
            "message": (
                "Rain prediction is available, but air quality "
                "prediction is not available for this date."
            )
        }

    return {
        "status": "success",
        "prediction_date": str(row[0]),
        "rain_probability": float(row[1]),
        "predicted_rain": int(row[2]),
        "air_quality_available": True,
        "aq_score_average": float(row[3]),
        "aq_score_min": float(row[4]),
        "aq_score_max": float(row[5]),
        "pollution_severity_average": float(row[6]),
        "pollution_severity_max": float(row[7])
    }

# Nova tool configuration
tool_config = {
    "tools": [{
        "toolSpec": {
            "name": "get_city_prediction",
            "description": (
                "Gets the Smart City prediction for a specific date "
                "from Exasol. The prediction contains both Random "
                "Forest rain prediction and daily air quality prediction."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "prediction_date": {
                            "type": "string",
                            "description": (
                                "The date for which the user wants "
                                "the Smart City prediction in YYYY-MM-DD format."
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
            "name": "get_city_prediction"
        }
    }
}

# Final Nova tool configuration
tool_config_final = {
    "tools": tool_config["tools"]
}

# Process user request
def process_request(user_input):
    messages = [{
        "role": "user",
        "content": [{
            "text": f"""
You are the Smart City AI assistant.

The user will provide a date and information about their plans,
travel schedule, walking, or outdoor activities.

Your job is to:

1. Extract the requested date from the user's message.
2. Convert the date to YYYY-MM-DD format.
3. MUST use the get_city_prediction tool for that date.
4. Use the returned rain and air-quality predictions.
5. Give ONLY the final natural-language answer to the user.

Do not describe your reasoning.
Do not mention the tool.
Do not mention Exasol.
Do not explain how the date was extracted.
Do not repeat the user's question.
Do not invent weather or air-quality information.

For rain, consider:
- rain probability
- whether rain is predicted
- the user's schedule
- walking or outdoor activities

For air quality, consider:
- average predicted AQ score
- minimum and maximum predicted AQ score
- average pollution severity
- maximum pollution severity

If air quality data is unavailable for the requested date,
clearly say that air-quality prediction is unavailable,
but still provide the rain-based recommendation.

Give concise, practical advice.

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

    # Handle tool call
    tool_called = False

    for content in output_message["content"]:
        if "toolUse" not in content:
            continue

        tool_called = True
        tool_use = content["toolUse"]

        result = get_city_prediction(
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

    # Handle missing tool call
    if not tool_called:
        for content in output_message["content"]:
            if "text" in content:
                return content["text"].strip()

        return "Unable to obtain a Smart City prediction."

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

    # Get final response
    final_text = ""

    for content in final_response["output"]["message"]["content"]:
        if "text" in content:
            final_text += content["text"]

    return final_text.strip()

# Theme colors
if st.session_state.dark_mode:
    bg = "#05080d"
    card = "#101820"
    input_bg = "#080d14"
    text = "#ffffff"
    secondary = "#8d9aaa"
    border = "#273545"
    result_border = "rgba(66, 165, 245, 0.28)"
    placeholder = "#657384"
else:
    bg = "#f4f7fb"
    card = "#ffffff"
    input_bg = "#ffffff"
    text = "#17212b"
    secondary = "#64748b"
    border = "#d6dee8"
    result_border = "rgba(25, 118, 210, 0.25)"
    placeholder = "#8a96a3"

# Frontend styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(35, 92, 150, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(0, 180, 216, 0.08),
                transparent 25%
            ),
            {bg};
        color: {text};
    }}

    .main .block-container {{
        max-width: 850px;
        padding-top: 25px;
        padding-bottom: 60px;
    }}

    .brand {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin-bottom: 8px;
    }}

    .brand-mark {{
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4fc3f7, #1976d2);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 30px rgba(33, 150, 243, 0.25);
        font-size: 21px;
    }}

    .brand-text {{
        font-size: 34px;
        font-weight: 750;
        letter-spacing: -1px;
        color: {text};
    }}

    .tagline {{
        text-align: center;
        color: {secondary};
        font-size: 16px;
        margin-bottom: 35px;
    }}

    .hero-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 22px;
        padding: 28px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.12);
        margin-bottom: 20px;
    }}

    .hero-title {{
        font-size: 22px;
        font-weight: 650;
        color: {text};
        margin-bottom: 7px;
    }}

    .hero-description {{
        color: {secondary};
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 22px;
    }}

    label {{
        color: {text} !important;
        font-weight: 550 !important;
    }}

    textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        border-radius: 14px !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 14px !important;
    }}

    textarea:focus {{
        border-color: #42a5f5 !important;
        box-shadow: 0 0 0 1px #42a5f5 !important;
    }}

    textarea::placeholder {{
        color: {placeholder} !important;
    }}

    .stButton {{
        margin-top: 14px;
    }}

    .stButton > button {{
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #42a5f5, #1976d2);
        color: #ffffff;
        font-size: 16px;
        font-weight: 650;
        box-shadow: 0 8px 25px rgba(25, 118, 210, 0.22);
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        background: linear-gradient(135deg, #64b5f6, #2196f3);
        color: #ffffff;
    }}

    .result-card {{
        background: {card};
        color: {text};
        border: 1px solid {result_border};
        border-radius: 20px;
        padding: 26px 28px;
        margin-top: 25px;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.12);
    }}

    .result-header {{
        color: #42a5f5;
        font-size: 13px;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 14px;
    }}

    .result-content {{
        color: {text};
        font-size: 17px;
        line-height: 1.75;
    }}

    .result-content p {{
        color: {text} !important;
    }}

    .footer {{
        text-align: center;
        color: {secondary};
        font-size: 12px;
        margin-top: 35px;
    }}

    .stSpinner > div {{
        border-top-color: #42a5f5 !important;
    }}

    .stAlert {{
        background-color: {card} !important;
        color: {text} !important;
        border-radius: 12px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Theme switch
theme_left, theme_right = st.columns([9, 1])

with theme_right:
    theme_label = "☀️" if st.session_state.dark_mode else "🌙"

    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Header
st.markdown(
    """
    <div class="brand">
        <div class="brand-text">UrbanSense AI</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="tagline">'
    'Intelligent weather and air-quality insights '
    'for smarter daily decisions'
    '</div>',
    unsafe_allow_html=True
)

# Input card
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">
            Plan your day
        </div>
        <div class="hero-description">
            Tell us where you are going, when you are travelling,
            and what outdoor activities you have planned.
            Our AI will check the weather and air-quality
            predictions and help you prepare.
        </div>
    """,
    unsafe_allow_html=True
)

user_input = st.text_area(
    "Your plans",
    placeholder=(
        "I'm going to college on 2023-06-16. "
        "I leave home at 6 AM and return at 7 PM. "
        "I have to walk for 20 minutes. "
        "Tell me if I should prepare for rain and "
        "whether the air quality will be suitable."
    ),
    height=145
)

# Check prediction
if st.button("Check My Day", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter your plans.")
    else:
        with st.spinner(
            "Analysing your plans and checking "
            "weather and air quality..."
        ):
            try:
                answer = process_request(user_input)

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-header">
                            Smart City Recommendation
                        </div>
                        <div class="result-content">
                            {answer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(
                    f"Unable to process the request: {e}"
                )
