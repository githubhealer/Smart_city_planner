import os
import boto3
import pyexasol
from dotenv import load_dotenv

load_dotenv()

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

# Get rain prediction from Exasol
def get_rain_prediction(prediction_date):
    conn = get_exasol_connection()

    try:
        parts = prediction_date.split("-")
        if len(parts) != 3:
            raise ValueError

        year, month, day = parts
        if len(year) != 4 or len(month) != 2 or len(day) != 2:
            raise ValueError

    except Exception:
        conn.close()
        return {
            "status": "error",
            "message": "Invalid date format. Use YYYY-MM-DD."
        }

    query = f"""
    SELECT OBS_DATE_OUT, RAIN_PROBABILITY, PREDICTED_RAIN
    FROM MILAN_WEATHER.MILAN_RF_PREDICTIONS
    WHERE OBS_DATE_OUT = DATE '{prediction_date}'
    """

    result = conn.execute(query)
    row = result.fetchone()
    conn.close()

    if row is None:
        return {
            "status": "error",
            "message": f"No prediction is available for {prediction_date}."
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
                "Gets the Random Forest rain prediction from the "
                "Exasol Smart City weather model for a specific date. "
                "Use this tool whenever the user asks about rain or "
                "weather risk for a specific date."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "prediction_date": {
                            "type": "string",
                            "description": (
                                "The date for which the user wants "
                                "the rain prediction in YYYY-MM-DD format."
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

# Configuration for final Nova call
tool_config_final = {
    "tools": tool_config["tools"]
}

# Get user input
print("\n==============================================")
print("       MILAN SMART CITY RAIN ASSISTANT")
print("==============================================\n")
print("Enter your request.\n")
print("Example:")
print(
    "I am going to college on 2023-06-16. "
    "I leave home at 6 AM and return at 7 PM. "
    "I have to walk for 20 minutes. "
    "Tell me if I should prepare for rain."
)
print("\nType 'exit' to quit.\n")

user_input = input("You: ")

if user_input.lower().strip() == "exit":
    print("Goodbye!")
    exit()

# Initial message
messages = [{
    "role": "user",
    "content": [{
        "text": f"""
You are a Smart City weather assistant.

The user will describe their plans and provide a date.

Your job is to:
1. Extract the date from the user's request.
2. Convert it to YYYY-MM-DD format.
3. Use the get_rain_prediction tool to obtain the Random Forest prediction.
4. After receiving the prediction, provide ONLY the final answer.
5. Give a concise, practical answer based on the prediction and user's plans.
6. Consider the user's schedule, walking, outdoor activities and other relevant details.

Do not describe your reasoning.
Do not mention tool calls.
Do not explain how you extracted the date.
Do not repeat the user's question.
Do not invent weather information.
Use only the prediction returned by the tool.

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

# Process tool call
tool_called = False

for content in output_message["content"]:
    if "toolUse" not in content:
        continue

    tool_called = True
    tool_use = content["toolUse"]

    print("\n----------------------------------------------")
    print("Nova requested tool:")
    print("----------------------------------------------")
    print("Tool:", tool_use["name"])
    print("Input:", tool_use["input"])

    result = get_rain_prediction(
        tool_use["input"]["prediction_date"]
    )

    print("\n----------------------------------------------")
    print("Exasol prediction:")
    print("----------------------------------------------")
    print(result)

    messages.append({
        "role": "user",
        "content": [{
            "toolResult": {
                "toolUseId": tool_use["toolUseId"],
                "content": [{"json": result}]
            }
        }]
    })

# Safety check
if not tool_called:
    print("\n----------------------------------------------")
    print("Nova did not call the prediction tool.")
    print("----------------------------------------------")

    for content in output_message["content"]:
        if "text" in content:
            print(content["text"])

    exit()

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

# Display final response
print("\n==============================================")
print("          SMART CITY RESPONSE")
print("==============================================\n")

for content in final_response["output"]["message"]["content"]:
    if "text" in content:
        text = content["text"]
        text = text.replace("<thinking>", "")
        text = text.replace("</thinking>", "")
        text = text.replace("<response>", "")
        text = text.replace("</response>", "")
        print(text.strip())
