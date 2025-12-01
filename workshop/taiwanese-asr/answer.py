import os
import io
import time
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess
from dotenv import load_dotenv

# ---------------------------
# 讀取 .env
# ---------------------------
load_dotenv()
YATING_API_KEY = os.getenv("YATING_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not YATING_API_KEY:
    raise RuntimeError("❌ 缺少 YATING_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ 缺少 GEMINI_API_KEY")


# ---------------------------
# 錄音設定
# ---------------------------
SR = 16000
CH = 1
DTYPE = "int16"


def record_audio():
    print("🎤 按 Enter 開始錄音，再按 Enter 停止")
    input()
    print("🎙️ 錄音中…")

    frames = []

    def cb(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SR, channels=CH, dtype=DTYPE, callback=cb)
    stream.start()
    input()
    stream.stop()
    stream.close()

    audio = np.concatenate(frames, axis=0)

    # --- 寫成 WAV（記憶體內） ---
    with io.BytesIO() as buf:
        sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

    # --- 寫入檔案：test.wav（方便 debug） ---
    with open("test.wav", "wb") as f:
        f.write(wav_bytes)
    print("💾 已輸出 test.wav，可用 Audacity/VLC 檢查錄音內容")

    return wav_bytes


# ---------------------------
# ffmpeg 修尾端靜音 + 轉 16kHz/mono
# ---------------------------
def ffmpeg_to_wav16k_mono(raw_bytes: bytes) -> bytes:
    p = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0",
         "-af", "silenceremove=stop_periods=-1:stop_threshold=-30dB:stop_duration=0.18",
         "-ar", "16000", "-ac", "1",
         "-f", "wav", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    out, _ = p.communicate(input=raw_bytes)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed")
    return out


# ---------------------------
# Yating ASR 主流程
# ---------------------------
def yating_asr_from_wav16k(wav16k_bytes: bytes):
    from ailabs_asr.streaming import StreamingClient

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav16k_bytes)
        f.flush()
        tmp = f.name

    state = {"result": None, "done": False}

    def on_processing_sentence(msg):
        txt = (msg.get("asr_sentence") or "").strip()
        if txt:
            state["result"] = txt
        print(f"（ASR partial）{txt}")

    def on_final_sentence(msg):
        txt = (msg.get("asr_sentence") or "").strip()
        if txt:
            state["result"] = txt
        state["done"] = True
        print(f"（ASR final）{txt}")

    def worker():
        cli = StreamingClient(key=YATING_API_KEY)
        cli.start_streaming_wav(
            pipeline="asr-zh-tw-std",
            file=tmp,
            on_processing_sentence=on_processing_sentence,
            on_final_sentence=on_final_sentence
        )

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    # 等 Final 事件
    while not state["done"]:
        time.sleep(0.05)

    try:
        os.unlink(tmp)
    except:
        pass

    return state["result"]


# ---------------------------
# Gemini 模型（Streaming 回覆）
# ---------------------------
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


def gemini_reply(text: str):
    print("\n🤖 Gemini 回覆：\n")
    final_text = ""
    for chunk in model.generate_content(text, stream=True):
        piece = chunk.text
        print(piece, end="", flush=True)
        final_text += piece
    print("\n")
    return final_text


# ---------------------------
# 主流程
# ---------------------------
def main():
    print("🌿 樹洞台語 ASR + Gemini Demo")

    # Step 1: 錄音
    wav = record_audio()

    # Step 2: ffmpeg
    print("\n⚙️ ffmpeg 轉換中…")
    wav16k = ffmpeg_to_wav16k_mono(wav)

    # Step 3: ASR
    print("🌀 Yating ASR 辨識中…")
    text = yating_asr_from_wav16k(wav16k)

    print("\n===== 🎧 ASR 辨識結果 =====")
    print(text)
    print("===========================\n")

    # Step 4: Gemini 對話
    gemini_reply(text)


if __name__ == "__main__":
    main()
