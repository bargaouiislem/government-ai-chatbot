import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = Client(host=OLLAMA_HOST)

# =========================
# LOAD MODELS & DATA
# =========================

print("🔄 تحميل الموديل...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("📂 تحميل البيانات...")

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

df = data["dataframe"]
embeddings = data["embeddings"]
documents = data["documents"]

print("✅ تم تحميل كل شيء بنجاح!\n")


# =========================
# SEARCH FUNCTION (Improved)
# =========================
def search(query, top_k=2, threshold=0.3):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    similarities = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # فلترة النتائج الضعيفة
    filtered_indices = [
        i for i in top_indices if similarities[i] > threshold
    ]

    if not filtered_indices:
        return None

    results = df.iloc[filtered_indices]

    # عرض المصادر
    if "source_table" in results.columns:
        print("\n📄 المصادر:")
        print(results["source_table"].values)

    return results, filtered_indices


# =========================
# LLM FUNCTION (Strict RAG)
# =========================
def ask_llama(context, question):
    chunk_size = 600
    context = context[:2000]

    chunks = [
        context[i:i + chunk_size]
        for i in range(0, len(context), chunk_size)
    ]

    for chunk in chunks:
        prompt = f"""
أنت مساعد إداري تونسي.

⚠️ أجب بالعربية فقط
⚠️ لا تخمّن
⚠️ إذا لا توجد معلومات قل: لا توجد معلومات كافية

المعلومات:
{chunk}

السؤال:
{question}

الإجابة:
"""

        try:
            response = client.chat(
                model="llama3:8b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )

            return response["message"]["content"]

        except Exception as e:
            print("⚠️ إعادة المحاولة...", e)

    return "لا توجد معلومات كافية"


# =========================
# BUILD CONTEXT FUNCTION
# =========================
def build_context(indices):
    return "\n\n".join([documents[i] for i in indices])


# =========================
# CHAT LOOP
# =========================
print("🤖 مرحبًا! أنا مساعدك الذكي 🤖")
print("يمكنك طرح أسئلة حول الخدمات الإدارية.")
print("اكتب 'خروج' لإنهاء المحادثة\n")

while True:
    query = input("🧑‍💻 أنت: ").strip()

    # تحقق من الإدخال
    if not query:
        print("⚠️ الرجاء كتابة سؤال.")
        continue

    if query in ["خروج", "انهاء", "إنهاء"]:
        print("👋 إلى اللقاء!")
        break

    # 🔍 SEARCH
    result = search(query)

    if result is None:
        print("\n🤖 المساعد:")
        print("لا توجد معلومات كافية في قاعدة البيانات.")
        continue

    results, indices = result

    # 🧠 BUILD CONTEXT
    context = build_context(indices)

    # 🤖 ASK MODEL
    answer = ask_llama(context, query)

    # 📢 DISPLAY
    print("\n🤖 المساعد:")
    print(answer)