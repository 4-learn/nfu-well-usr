import os
from dotenv import load_dotenv
import google.generativeai as genai

# 讀取 .env 檔
load_dotenv()

# 從環境變數取得 API key
api_key = os.getenv("GEMINI_API_KEY")

# 檢查 key 是否存在
if not api_key:
    raise ValueError("❌ 找不到 GEMINI_API_KEY，請確認 .env 檔是否存在並有內容。")

# 設定 Gemini API key
genai.configure(api_key=api_key)

# 初始化模型
model = genai.GenerativeModel("gemini-2.5-flash")

# 發送 prompt

response = model.generate_content("保安宮拜什麼神？")

# 輸出結果
print("🤖 Gemini 回覆：", response.text)