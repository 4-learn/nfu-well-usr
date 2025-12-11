import yaml
import os
from dotenv import load_dotenv
import google.generativeai as genai

# === 1. 初始化 ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === 感測器假資料 ===
def get_sensor_data():
    with open("sensor.yaml", "r") as f:
        data = yaml.safe_load(f)
    print("[get_sensor_data]", data)
    return data

# === MQTT 模擬 ===
def send_mqtt_command(topic: str, payload: str):
    print(f"[MQTT] 發送 → {topic} = {payload}")
    return {"status": "sent", "topic": topic, "payload": payload}

# === 記錄事件 ===
def log_event(message: str):
    with open("events.log", "a") as f:
        f.write(message + "\n")
    print("[log_event] 已記錄：", message)
    return {"saved": True}

# === 工具宣告 ===
tools = [
    {
        "function_declarations": [
            {
                "name": "get_sensor_data",
                "description": "讀取感測器資料 (salinity, ph, do, temp)",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "send_mqtt_command",
                "description": "發送 MQTT 控制指令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "payload": {"type": "string"},
                    },
                    "required": ["topic", "payload"]
                }
            },
            {
                "name": "log_event",
                "description": "記錄 AI 的決策事件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            },
        ]
    }
]

# === 建立模型 ===
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    tools=tools,
)

chat = model.start_chat(history=[])

# === 處理工具呼叫 ===
def handle_tool_calls(resp):
    for part in resp.candidates[0].content.parts:
        if getattr(part, "function_call", None):
            fname = part.function_call.name
            args = dict(part.function_call.args)

            print(f"🤖 呼叫函式：{fname} {args}")

            if fname == "get_sensor_data":
                result = get_sensor_data()
            elif fname == "send_mqtt_command":
                result = send_mqtt_command(args["topic"], args["payload"])
            elif fname == "log_event":
                result = log_event(args["message"])
            else:
                result = {"error": "未知函式"}

            tool_msg = {
                "role": "tool",
                "parts": [
                    {"function_response": {"name": fname, "response": result}}
                ]
            }

            follow = chat.send_message(tool_msg)
            return follow

    return resp

# === 主互動 ===
print("水井村智慧養殖 Agent（輸入 exit 離開）\n")

while True:
    q = input("你說：")
    if q == "exit": break

    resp = chat.send_message(q)
    final = handle_tool_calls(resp)
    print("💬 最終回答：", final.text)
