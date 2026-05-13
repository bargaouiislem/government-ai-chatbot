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
    all_docs = data["documents"]
    texts = data["texts"]
else:
    texts = data["documents"]
    all_docs = [{"text": t, "source_table": "", "procedure": ""} for t in texts]

embeddings = data["embeddings"]

print(f"✅ تم تحميل {len(texts)} وثيقة بنجاح!\n")


# =========================
# SEARCH FUNCTION
# =========================
def search(query, top_k=10, threshold=0.20):
    """
    Search for most relevant documents.
    - top_k increased to 10 for better synonym recall
    - threshold lowered slightly to catch paraphrase matches
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
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        print(f"\n⚠️ أفضل نتيجة: {best_score:.3f} (أقل من العتبة {threshold})")
        if best_score > 0.12:
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
    """
    Build a well-structured, complete context from search results.
    Priority: full procedure documents first, then sections, then QA fallback.
    """
    seen_procedures = set()
    parts = []

    # Priority 1: full procedure documents
    for idx, score in filtered_results:
        doc = all_docs[idx]
        proc = doc.get("procedure", "")
        section = doc.get("section", "")
        text = doc.get("text", texts[idx])

        if section == "كل المعلومات" and proc not in seen_procedures:
            seen_procedures.add(proc)
            parts.append(f"=== معلومات كاملة عن إجراء: {proc} ===\n{text}")

    # Priority 2: section docs for procedures not yet covered
    for idx, score in filtered_results:
        doc = all_docs[idx]
        proc = doc.get("procedure", "")
        section = doc.get("section", "")
        text = doc.get("text", texts[idx])

        if section not in ("كل المعلومات", "سؤال وجواب") and proc not in seen_procedures:
            seen_procedures.add(proc)
            parts.append(f"=== {proc} - {section} ===\n{text}")

    # Priority 3: QA fallback
    if not parts:
        for idx, score in filtered_results:
            doc = all_docs[idx]
            proc = doc.get("procedure", "")
            section = doc.get("section", "")
            text = doc.get("text", texts[idx])

            if section == "سؤال وجواب" and proc not in seen_procedures:
                seen_procedures.add(proc)
                parts.append(f"=== {proc} ===\n{text}")

    # Hard fallback
    if not parts and filtered_results:
        idx, score = filtered_results[0]
        parts.append(all_docs[idx].get("text", texts[idx]))

    return "\n\n---\n\n".join(parts)


# =========================
# LLM FUNCTION
# =========================
def ask_llama(context, question):
    """
    Send question + context to Llama and return answer in Arabic only.
    """
    context = context[:5000]

    prompt = f"""أنت مساعد إداري رسمي تابع لوزارة التجارة وتنمية الصادرات التونسية.
مهمتك الوحيدة هي الإجابة على أسئلة المواطنين بالعربية فقط، بناءً حصراً على المعلومات المقدمة في السياق أدناه.

══════════════════════════════════════
قواعد صارمة ومطلقة — لا استثناء:
══════════════════════════════════════
١. اللغة: العربية فقط وحصراً في كل كلمة من إجابتك.
   - ممنوع تماماً استخدام الفرنسية أو الإنجليزية أو أي لغة أخرى (صينية، روسية، إلخ).
   - حتى لو كان السؤال بلغة أخرى، أجب بالعربية فقط.
   - إذا وجدت كلمات أجنبية في السياق، ترجمها أو أهملها ولا تعيد كتابتها.

٢. المصدر: استخدم فقط المعلومات الموجودة في السياق أدناه.
   - لا تخترع أي معلومة.
   - لا تخمّن أي معلومة غير موجودة في السياق.
   - إذا لم تجد الإجابة في السياق، قل بالضبط: "لا توجد معلومات كافية حول هذا الموضوع في قاعدة بيانات الوزارة."

٣. التنظيم: قدّم إجابة منظّمة وواضحة:
   - ابدأ بذكر اسم الإجراء.
   - استخدم نقاطاً أو أرقاماً عند الحاجة.
   - كن دقيقاً وشاملاً، لا تحذف أي معلومة مهمة من السياق.

٤. التعامل مع المرادفات: إذا سأل المواطن عن "الوثائق" أو "الأوراق" أو "المستندات" أو "الملفات" أو "ما يلزم" — فهو يسأل عن نفس الشيء (الوثائق المطلوبة). تصرّف بحسب ذلك.

══════════════════════════════════════
المعلومات المتاحة من قاعدة بيانات وزارة التجارة وتنمية الصادرات:
══════════════════════════════════════
{context}

══════════════════════════════════════
سؤال المواطن:
══════════════════════════════════════
{question}

══════════════════════════════════════
الإجابة (بالعربية فقط، لا تكتب أي حرف بغير العربية):
══════════════════════════════════════"""

    try:
        response = client.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,   # fully deterministic — no hallucination
                "num_predict": 512,
            }
        )
        if isinstance(response, dict):
            answer = response["message"]["content"]
        else:
            answer = response.message.content

        # Safety check: warn if response has almost no Arabic characters
        arabic_chars = sum(1 for c in answer if '\u0600' <= c <= '\u06FF')
        if arabic_chars < 10:
            return "حدث خطأ في معالجة الإجابة. يرجى إعادة صياغة سؤالك بالعربية والمحاولة مجدداً."

        return answer

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