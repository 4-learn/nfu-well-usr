import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ============================================
#  0. 初始化：讀取 .env
# ============================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ 找不到 GEMINI_API_KEY，請確認 .env 是否正確")

genai.configure(api_key=api_key)

print("🔑 GEMINI_API_KEY 已成功載入")


# ============================================
#  1. 載入 FAQ
# ============================================
docs = []
with open("faq_aquaculture.txt", "r", encoding="utf-8") as f:
    chunks = f.read().strip().split("\n\n")
    for chunk in chunks:
        docs.append(chunk.strip())

print(f"📄 已載入 FAQ 筆數：{len(docs)}")


# ============================================
#  2. 舊版 embed API
# ============================================
def embed_text(text: str):
    """使用 embed_content()（舊 SDK API）"""
    res = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(res["embedding"])


print("⚙️ 正在建立 FAQ 向量（embedding）...")
embeddings = [embed_text(doc) for doc in docs]
print("✅ FAQ 向量化完成")


# ============================================
#  3. Stage 1：Embedding 粗篩
# ============================================
def search_similar(query, top_k=3):

    print("\n============================================================")
    print(f"🔍 使用者問題：{query}")
    print("============================================================")

    q_emb = embed_text(query)

    sims = []
    for idx, d_emb in enumerate(embeddings):
        sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
        sims.append(sim)

    # 從大到小排序
    sorted_indices = np.argsort(sims)[::-1][:top_k]

    print("\n【階段 1】Embedding 相似度粗篩結果：\n")
    for rank, i in enumerate(sorted_indices, 1):
        print(f"  第 {rank} 名：{sims[i]:.3f} → {docs[i][:50]}...")

    return sorted_indices, [docs[i] for i in sorted_indices]


# ============================================
#  4. Stage 2：Reranker
# ============================================
def rerank(query, candidates):

    print("\n【階段 2】Reranker 精準排序\n")
    scores = []

    model = genai.GenerativeModel("gemini-2.5-flash")

    for idx, doc in enumerate(candidates):

        prompt = f"""
你是一位文蛤養殖技術評估員，任務是判斷 FAQ 是否能幫助回答問題。

【問題】
{query}

【FAQ】
{doc}

請依以下標準，回傳 0–100 的整數分數：
- 100：FAQ 完整回答問題
- 70–99：FAQ 提供高度相關資訊
- 40–69：FAQ 提到部分相關主題
- 1–39：FAQ 幾乎不相關，但略有語意接近
- 0：完全無關

請「只回傳數字」。"""

        response = model.generate_content(prompt)
        raw = (response.text or "").strip()

        # 萃取分數
        digits = "".join(ch for ch in raw if ch.isdigit())
        score = int(digits) if digits else 0

        scores.append(score)
        print(f"候選 {idx+1}：{score} 分 → {doc[:50]}...\n")

    return scores


# ============================================
#  5. Stage 3：最終回答（強化推論版）
# ============================================
def generate_final_answer(query, best_doc):

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
你是一位文蛤養殖專家。

你會收到 FAQ（系統選出的參考資訊），但 FAQ 內容可能不完整。
請依照下列規則產生答案：

⚠️ 規則（非常重要）：
1. FAQ 內容只是參考，不是限制。
2. FAQ 若沒有明確答案，你必須根據你的水產養殖知識與常識進行推論。
3. 絕對不能回答「FAQ 沒提到」「查無資料」。
4. 必須給出學生可以理解的、合理且具可操作性的建議。
5. 回答要簡潔、可信、具教育性。

【FAQ（參考）】
{best_doc}

【問題】
{query}

請給出你的最終答案：
"""

    response = model.generate_content(prompt)
    return response.text


# ============================================
#  6. RAG 主流程
# ============================================
def answer_with_rag(query):

    # Stage 1：Embedding 搜尋
    indices, candidates = search_similar(query)

    # Stage 2：Reranker
    scores = rerank(query, candidates)

    # 結合排序
    combined = list(zip(indices, candidates, scores))
    combined.sort(key=lambda x: x[2], reverse=True)

    best_doc = combined[0][1]

    print("\n🏆【最終選中 FAQ】\n")
    print(best_doc)
    print("\n============================================================")

    print("🤖 正在產生最終回答...\n")
    answer = generate_final_answer(query, best_doc)

    print(answer)
    print("============================================================\n")


# ============================================
#  7. CLI
# ============================================
if __name__ == "__main__":
    while True:
        q = input("\n請輸入問題（exit 離開）：")
        if q.lower().strip() == "exit":
            break
        answer_with_rag(q)
