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
    df, embeddings = pickle.load(f)

# =========================
# Search function
# =========================
def search(query, top_k=3):
    query_embedding = model.encode([query])[0]

    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

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
    # Split context into chunks
    chunk_size = 1000  # characters per chunk
    chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]

    partial_answers = []

    for chunk in chunks:
        prompt = f"""
أجب باللغة العربية الرسمية.

اعتمد فقط على المعطيات التالية.
أجب بجزء من الإجابة فقط بناءً على هذا الجزء.

المعطيات:
{chunk}

السؤال:
{question}
"""

        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
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
        model="llama3",
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
    context = "\n".join(
        [str(x) for x in results.values.flatten() if str(x) != "nan" and str(x).strip() != ""]
    )

    # Ask model
    answer = ask_llama(context, query)

    # Display answer
    print("\n🤖 المساعد:")
    print(answer)