import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# =========================
# Load embedding model
# =========================
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# =========================
# Load saved data
# =========================
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

    df = data["dataframe"]
    embeddings = data["embeddings"]
    documents = data["documents"]

# =========================
# Search function
# =========================
def search(query, top_k=3):
    query_embedding = model.encode(
    [query],
    normalize_embeddings=True
    )[0]

    similarities = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = df.iloc[top_indices]

    # ✅ Show sources (important for PFA)
    if "source_table" in results.columns:
        print("\n📄 المصادر:")
        print(results["source_table"].values)

    return results

# =========================
# LLaMA function
# =========================
def ask_llama(context, question):
    # تقسيم النص إلى أجزاء صغيرة
    chunk_size = 1200
    chunks = [
        context[i:i + chunk_size]
        for i in range(0, len(context), chunk_size)
    ]

    partial_answers = []

    for chunk in chunks:
        prompt = f"""
أنت مساعد إداري تونسي.

أجب فقط اعتماداً على المعلومات التالية.
إذا لم تجد الإجابة قل: لا توجد معلومات كافية.

المعلومات:
{chunk}

السؤال:
{question}

الإجابة:
"""

        try:
            response = ollama.chat(
                model="llama3:8b",
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2
                }
            )

            partial_answers.append(response["message"]["content"])

        except Exception as e:
            print("⚠️ خطأ في جزء من المعالجة:", e)

    # Final merge step
    final_prompt = f"""
قم بدمج الإجابات التالية في إجابة واحدة واضحة ومنظمة بدون تكرار:

{chr(10).join(partial_answers)}
"""

    final_response = ollama.chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": final_prompt}]
    )

    return final_response["message"]["content"]

# =========================
# Chat loop (Arabic UI)
# =========================
print("🤖 مرحبًا! أنا مساعدك الذكي 🤖")
print("يمكنك طرح أي سؤال حول الخدمات الإدارية.")
print("لإنهاء المحادثة، اكتب: خروج\n")

while True:
    query = input("🧑‍💻 أنت: ").strip()

    # Empty input check
    if query == "":
        print("⚠️ الرجاء كتابة سؤال.")
        continue

    # Exit conditions
    if query in ["خروج", "انهاء", "إنهاء"]:
        print("👋 إلى اللقاء! سعدت بمساعدتك.")
        break

    # Search
    results = search(query)

    # Clean context (remove NaN and empty values)
    context = "\n\n".join(
    [documents[i] for i in results.index]
    )

    # Ask model
    answer = ask_llama(context, query)

    # Display answer
    print("\n🤖 المساعد:")
    print(answer)