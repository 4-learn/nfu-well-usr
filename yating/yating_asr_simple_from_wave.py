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

# -------------------------------------------------------
# 讀取 YATING_API_KEY
# -------------------------------------------------------
load_dotenv()
YATING_API_KEY = os.getenv("YATING_API_KEY", "").strip()

if not YATING_API_KEY:
    raise RuntimeError("❌ 請先設定環境變數 YATING_API_KEY")


# -------------------------------------------------------
# 錄音設定
# -------------------------------------------------------
SR = 16000
CH = 1
DTYPE = "int16"


def record_audio():
    print("📂 從 test.wav 讀取音檔…")

    filename = "test.wav"
    if not os.path.exists(filename):
        raise RuntimeError("❌ 找不到 test.wav，請先放入同目錄")

    # 讀取 WAV → numpy 陣列 + 取樣率
    data, sr = sf.read(filename, dtype="int16")

    # 若不是 16kHz/mono，後面 ffmpeg 會轉，不影響
    print(f"📄 test.wav 讀取成功，形狀={data.shape}, SR={sr}")

    # 轉成 bytes（WAV 格式）
    with io.BytesIO() as buf:
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

    return wav_bytes


# -------------------------------------------------------
# ffmpeg：轉 16kHz / mono / 去尾端靜音
# -------------------------------------------------------
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


# -------------------------------------------------------
# Yating ASR 主流程
# -------------------------------------------------------
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

    # 等結束
    while not state["done"]:
        time.sleep(0.05)

    try:
        os.unlink(tmp)
    except:
        pass

    return state["result"]


# -------------------------------------------------------
# 主流程：錄音 → ASR → 印出文字
# -------------------------------------------------------
def main():
    print("🌿 Yating 台語 ASR Demo 版")
    wav = record_audio()

    print("\n⚙️ ffmpeg 轉換中…")
    wav16k = ffmpeg_to_wav16k_mono(wav)

    print("🌀 Yating ASR 辨識中…")
    text = yating_asr_from_wav16k(wav16k)

    print("\n===== 🎧 ASR 辨識結果 =====")
    print(text)
    print("===========================\n")


if __name__ == "__main__":
    main()
