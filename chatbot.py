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
 
# Support both old and new embeddings.pkl format
if "documents" in data and isinstance(data["documents"][0], dict):
    # New format: list of dicts
    all_docs = data["documents"]
    texts = data["texts"]
else:
    # Old format: flat list of strings
    texts = data["documents"]
    all_docs = [{"text": t, "source_table": "", "procedure": ""} for t in texts]
 
embeddings = data["embeddings"]
 
print(f"✅ تم تحميل {len(texts)} وثيقة بنجاح!\n")
 
 
# =========================
# SEARCH FUNCTION
# =========================
def search(query, top_k=5, threshold=0.25):
    """
    Search for most relevant documents.
    - top_k: how many top results to retrieve
    - threshold: minimum similarity score (lowered to 0.25 for Arabic)
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]
 
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
 
    # Filter by threshold
    filtered = [(i, similarities[i]) for i in top_indices if similarities[i] > threshold]
 
    if not filtered:
        # If nothing passes threshold, return best match anyway with a warning
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        print(f"\n⚠️ أفضل نتيجة: {best_score:.3f} (أقل من العتبة {threshold})")
        if best_score > 0.15:
            return [(best_idx, best_score)]
        return []
 
    print(f"\n📄 نتائج البحث (أفضل {len(filtered)}):")
    for idx, score in filtered:
        src = all_docs[idx].get("source_table", "")
        proc = all_docs[idx].get("procedure", "")
        print(f"  [{score:.3f}] {src} | {proc[:40]}")
 
    return filtered
 
 
# =========================
# BUILD CONTEXT
# =========================
def build_context(filtered_results):
    """Build a well-structured context from search results."""
    parts = []
    for idx, score in filtered_results:
        doc = all_docs[idx]
        src = doc.get("source_table", "")
        text = doc.get("text", texts[idx])
        parts.append(f"[مصدر: {src}]\n{text}")
    return "\n\n---\n\n".join(parts)
 
 
# =========================
# LLM FUNCTION
# =========================
def ask_llama(context, question):
    """
    Send question + context to Llama and return answer.
    Context is NOT chunked — we send it all at once.
    Max context limited to 4000 chars to stay within LLM limits.
    """
    # Trim context if very long, but keep more than before
    context = context[:4000]
 
    prompt = f"""أنت مساعد إداري متخصص في خدمات وزارة التجارة وتنمية الصادرات التونسية.
 
مهمتك: الإجابة على سؤال المواطن بناءً فقط على المعلومات المقدمة أدناه.
 
⚠️ قواعد مهمة:
- أجب بالعربية فقط
- استخدم فقط المعلومات المذكورة في السياق
- إذا لم تجد الجواب في السياق، قل: "لا توجد معلومات كافية حول هذا الموضوع"
- لا تخترع أو تخمّن أي معلومة
- كن واضحاً ومنظماً في إجابتك
 
=== المعلومات المتاحة ===
{context}
 
=== سؤال المواطن ===
{question}
 
=== الإجابة ==="""
 
    try:
        response = client.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 512,   # limit response length
            }
        )
        # Handle both dict and object response formats
        if isinstance(response, dict):
            return response["message"]["content"]
        else:
            return response.message.content
 
    except Exception as e:
        print(f"⚠️ خطأ في الاتصال بـ Ollama: {e}")
        return "حدث خطأ في الاتصال بالنموذج. يرجى المحاولة مجدداً."
 
 
# =========================
# CHAT LOOP
# =========================
print("=" * 50)
print("🤖 مرحبًا! أنا مساعدك في وزارة التجارة وتنمية الصادرات 🤖")
print("يمكنك طرح أسئلة حول الإجراءات والوثائق والخدمات الإدارية.")
print("اكتب 'خروج' لإنهاء المحادثة")
print("=" * 50 + "\n")
 
while True:
    query = input("🧑‍💻 أنت: ").strip()
 
    if not query:
        print("⚠️ الرجاء كتابة سؤال.")
        continue
 
    if query in ["خروج", "انهاء", "إنهاء", "exit", "quit"]:
        print("👋 إلى اللقاء!")
        break
 
    # 🔍 SEARCH
    results = search(query)
 
    if not results:
        print("\n🤖 المساعد:")
        print("لا توجد معلومات كافية في قاعدة البيانات حول هذا الموضوع.")
        print()
        continue
 
    # 🧠 BUILD CONTEXT
    context = build_context(results)
 
    # 🤖 ASK MODEL
    print("\n🤖 المساعد:")
    answer = ask_llama(context, query)
    print(answer)
    print()