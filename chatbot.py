import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = Client(host=OLLAMA_HOST)

# =========================
# GREETING DETECTION
# =========================
GREETING_PATTERNS = [
    r"^(أهلا|اهلا|مرحبا|مرحباً|السلام عليكم|سلام|صباح الخير|مساء الخير|هلا|يسلمو|يسلموا|شكراً|شكرا|وداعا|مع السلامة|إلى اللقاء)\b",
    r"^(hi|hello|hey|bonjour|salut|bonsoir|bye|goodbye|thanks|thank you|merci|ok|okay|yes|no|yep|nope|lol|haha)\b",
]

NONSENSE_MIN_LENGTH = 3
# Raised from 0.12 → 0.30 to prevent answering nonsense/off-topic queries
NONSENSE_THRESHOLD = 0.30

GREETING_RESPONSE = "أهلاً وسهلاً! 😊 كيف يمكنني مساعدتك اليوم؟ يمكنك سؤالي عن أي إجراء أو خدمة تقدمها وزارة التجارة وتنمية الصادرات."
NO_INFO_RESPONSE  = "لا توجد معلومات كافية حول هذا الموضوع في قاعدة بيانات الوزارة."


def is_greeting(text: str) -> bool:
    """Return True if the message is a greeting/small-talk and not a real question."""
    t = text.strip()
    if len(t) < NONSENSE_MIN_LENGTH:
        return True
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE | re.UNICODE):
            return True
    return False


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
    Returns empty list if best score is below NONSENSE_THRESHOLD
    to avoid answering random/off-topic inputs.
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    filtered = [(i, similarities[i]) for i in top_indices if similarities[i] > threshold]

    if not filtered:
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        print(f"\n⚠️ أفضل نتيجة: {best_score:.3f} (أقل من العتبة {threshold})")
        # ✅ FIX: raised floor from 0.12 → NONSENSE_THRESHOLD (0.30)
        if best_score > NONSENSE_THRESHOLD:
            return [(best_idx, best_score)]
        return []

    # ✅ FIX: discard all results if the best one doesn't reach NONSENSE_THRESHOLD
    best_score_in_results = max(score for _, score in filtered)
    if best_score_in_results < NONSENSE_THRESHOLD:
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
مهمتك الوحيدة هي الإجابة على سؤال المواطن الحالي فقط، بالعربية، بناءً حصراً على المعلومات المقدمة في السياق أدناه.

══════════════════════════════════════
قواعد صارمة ومطلقة — لا استثناء:
══════════════════════════════════════
١. اللغة: العربية فقط وحصراً في كل كلمة من إجابتك.
   - ممنوع تماماً استخدام الفرنسية أو الإنجليزية أو أي لغة أخرى.
   - حتى لو كان السؤال بلغة أخرى، أجب بالعربية فقط.

٢. المصدر: استخدم فقط المعلومات الموجودة في السياق أدناه.
   - لا تخترع أي معلومة ولا تخمّن.
   - إذا لم تجد الإجابة في السياق، قل بالضبط: "لا توجد معلومات كافية حول هذا الموضوع في قاعدة بيانات الوزارة."
   - لا تذكر أبداً أسماء الجداول أو قاعدة البيانات أو مصدر المعلومات. أجب مباشرة بالمعلومة فقط.

٣. السؤال الحالي فقط: أجب على سؤال المواطن الحالي فقط.
   - لا تتطوع بمعلومات عن إجراءات أخرى لم يسأل عنها.
   - لا تربط إجابتك بأي سؤال سابق في المحادثة.
   - كل سؤال مستقل بذاته.

٤. التنظيم: قدّم إجابة منظّمة وواضحة.
   - استخدم نقاطاً أو أرقاماً عند الحاجة.
   - كن دقيقاً وشاملاً، لا تحذف أي معلومة مهمة من السياق.

٥. التعامل مع المرادفات: إذا سأل المواطن عن "الوثائق" أو "الأوراق" أو "المستندات" أو "الملفات" أو "ما يلزم" — فهو يسأل عن نفس الشيء (الوثائق المطلوبة). تصرّف بحسب ذلك.

══════════════════════════════════════
المعلومات المتاحة:
══════════════════════════════════════
{context}

══════════════════════════════════════
سؤال المواطن الحالي:
══════════════════════════════════════
{question}

══════════════════════════════════════
الإجابة (بالعربية فقط، لا تذكر مصادر أو أسماء جداول):
══════════════════════════════════════"""

    try:
        response = client.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_predict": 512,
            }
        )
        if isinstance(response, dict):
            answer = response["message"]["content"]
        else:
            answer = response.message.content

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

    # ✅ FIX 1: Detect greetings BEFORE searching
    if is_greeting(query):
        print(f"\n🤖 المساعد:\n{GREETING_RESPONSE}\n")
        continue

    # ✅ FIX 2: search() now enforces NONSENSE_THRESHOLD
    results = search(query)

    if not results:
        print(f"\n🤖 المساعد:\n{NO_INFO_RESPONSE}\n")
        continue

    # ✅ FIX 3: ask_llama prompt now forbids sources and context bleeding
    context = build_context(results)
    print("\n🤖 المساعد:")
    answer = ask_llama(context, query)
    print(answer)
    print()