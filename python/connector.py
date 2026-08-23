import os
import streamlit as st
import boto3
import pyexasol
from dotenv import load_dotenv

load_dotenv()

# Configuration
st.set_page_config(
    page_title="Milan Smart City",
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

# Rain prediction
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

# Nova tool configuration
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

# Final Nova tool configuration
tool_config_final = {
    "tools": tool_config["tools"]
}

# Nova processing
def process_request(user_input):
    messages = [{
        "role": "user",
        "content": [{
            "text": f"""
You are a Smart City weather assistant.

The user will provide a date and information about their plans.

Extract the requested date and convert it to YYYY-MM-DD format.

You MUST use the get_rain_prediction tool for that date.

After receiving the prediction, provide ONLY the final answer to the user.

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
    tool_called = False

    for content in output_message["content"]:
        if "toolUse" not in content:
            continue

        tool_called = True
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

    if not tool_called:
        for content in output_message["content"]:
            if "text" in content:
                return content["text"].strip()

        return "Unable to obtain a weather prediction."

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

    # Get final response text
    final_text = ""

    for content in final_response["output"]["message"]["content"]:
        if "text" in content:
            final_text += content["text"]

    return final_text.strip()

# Theme values
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

    /* Theme button */
    .theme-container {{
        display: flex;
        justify-content: flex-end;
        margin-bottom: 5px;
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
        <div class="brand-mark">☁</div>
        <div class="brand-text">Milan Smart City</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="tagline">Intelligent weather insights for smarter daily decisions</div>',
    unsafe_allow_html=True
)

# Input card
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Plan your day</div>
        <div class="hero-description">
            Tell us where you are going, when you are travelling,
            and what outdoor activities you have planned.
            Our AI will check the weather prediction and help you prepare.
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
        "Tell me if I should prepare for rain."
    ),
    height=145
)

if st.button("Check My Day", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter your plans.")
    else:
        with st.spinner(
            "Analysing your plans and checking the prediction..."
        ):
            try:
                answer = process_request(user_input)

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-header">
                            Weather Recommendation
                        </div>
                        <div class="result-content">
                            {answer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Unable to process the request: {e}")
