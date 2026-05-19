import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = Client(host=OLLAMA_HOST)

# =========================
# GREETING / NONSENSE DETECTION
# =========================
GREETING_PATTERNS = [
    r"^(أهلا|اهلا|مرحبا|مرحباً|السلام عليكم|سلام|صباح الخير|مساء الخير|هلا|يسلمو|يسلموا|شكراً|شكرا|وداعا|مع السلامة|إلى اللقاء)\b",
    r"^(hi|hello|hey|bonjour|salut|bonsoir|bye|goodbye|thanks|thank you|merci|ok|okay|yes|no|yep|nope|lol|haha)\b",
]

NONSENSE_MIN_LENGTH = 4
NONSENSE_THRESHOLD = 0.38

GREETING_RESPONSE = "أهلاً وسهلاً! 😊 كيف يمكنني مساعدتك اليوم؟ يمكنك سؤالي عن أي إجراء أو خدمة تقدمها وزارة التجارة وتنمية الصادرات."
NO_INFO_RESPONSE  = "عذراً، لا توجد معلومات كافية حول هذا الموضوع."


def is_greeting(text: str) -> bool:
    t = text.strip()
    if len(t) < NONSENSE_MIN_LENGTH:
        return True
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE | re.UNICODE):
            return True
    return False


def is_nonsense(text: str) -> bool:
    t = text.strip()
    if len(t) < NONSENSE_MIN_LENGTH:
        return True
    arabic_chars = sum(1 for c in t if '\u0600' <= c <= '\u06FF')
    latin_chars  = sum(1 for c in t if c.isalpha() and c.isascii())
    total_chars  = len(t.replace(" ", ""))
    if total_chars > 0 and (arabic_chars + latin_chars) / total_chars < 0.4:
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
# SEARCH
# =========================
def search(query, top_k=10, threshold=0.20):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    best_score = float(similarities[top_indices[0]])

    # Hard gate: if best score is below NONSENSE_THRESHOLD → no answer
    if best_score < NONSENSE_THRESHOLD:
        print(f"\n⚠️ أفضل نتيجة: {best_score:.3f} — أقل من العتبة {NONSENSE_THRESHOLD} → لا إجابة")
        return []

    filtered = [(i, float(similarities[i])) for i in top_indices if similarities[i] > threshold]
    if filtered:
        print(f"\n📄 نتائج البحث ({len(filtered)}):")
        for idx, score in filtered[:3]:
            proc = all_docs[idx].get("procedure", "")
            print(f"  [{score:.3f}] {proc[:50]}")
    return filtered


# =========================
# BUILD CONTEXT
# =========================
def build_context(filtered_results):
    seen_procedures = set()
    parts = []

    for idx, score in filtered_results:
        doc = all_docs[idx]
        proc = doc.get("procedure", "")
        section = doc.get("section", "")
        text = doc.get("text", texts[idx])
        if section == "كل المعلومات" and proc not in seen_procedures:
            seen_procedures.add(proc)
            parts.append(f"=== معلومات كاملة عن إجراء: {proc} ===\n{text}")

    for idx, score in filtered_results:
        doc = all_docs[idx]
        proc = doc.get("procedure", "")
        section = doc.get("section", "")
        text = doc.get("text", texts[idx])
        if section not in ("كل المعلومات", "سؤال وجواب") and proc not in seen_procedures:
            seen_procedures.add(proc)
            parts.append(f"=== {proc} - {section} ===\n{text}")

    if not parts:
        for idx, score in filtered_results:
            doc = all_docs[idx]
            proc = doc.get("procedure", "")
            section = doc.get("section", "")
            text = doc.get("text", texts[idx])
            if section == "سؤال وجواب" and proc not in seen_procedures:
                seen_procedures.add(proc)
                parts.append(f"=== {proc} ===\n{text}")

    if not parts and filtered_results:
        idx, score = filtered_results[0]
        parts.append(all_docs[idx].get("text", texts[idx]))

    return "\n\n---\n\n".join(parts)


# =========================
# FIX BULLET POINTS → NUMBERED LIST
# =========================
def fix_numbering(text: str) -> str:
    lines = text.split("\n")
    result = []
    counter = 1
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("* ", "- ", "• ")):
            content = stripped[2:].strip()
            result.append(f"{counter}. {content}")
            counter += 1
        else:
            result.append(line)
            if stripped == "":
                counter = 1
    return "\n".join(result)


# =========================
# ASK LLAMA
# =========================
def ask_llama(context, question):
    context = context[:6000]

    prompt = f"""أنت مساعد إداري رسمي تابع لوزارة التجارة وتنمية الصادرات التونسية.

قواعد صارمة:
١. أجب بالعربية فقط — ممنوع أي لغة أخرى.
٢. استخدم فقط المعلومات الموجودة في السياق أدناه — لا تخترع أي معلومة.
٣. إذا لم تجد الإجابة، قل فقط: "عذراً، لا توجد معلومات كافية حول هذا الموضوع." ولا تضف أي شيء آخر.
٤. لا تذكر أسماء جداول أو قاعدة بيانات أو مصدر المعلومات.
٥. رقّم كل عنصر في القوائم بأرقام هكذا: 1. ثم 2. ثم 3. — ممنوع استخدام * أو - أو •
٦. أجب على السؤال الحالي فقط.

=== المعلومات المتاحة ===
{context}

=== سؤال المواطن ===
{question}

=== الإجابة (بالعربية، قوائم مرقّمة 1. 2. 3. فقط) ==="""

    try:
        response = client.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 1024}
        )
        if isinstance(response, dict):
            answer = response["message"]["content"]
        else:
            answer = response.message.content

        arabic_chars = sum(1 for c in answer if '\u0600' <= c <= '\u06FF')
        if arabic_chars < 10:
            return "عذراً، لا توجد معلومات كافية حول هذا الموضوع."

        answer = fix_numbering(answer)
        return answer.strip()

    except Exception as e:
        print(f"⚠️ خطأ في الاتصال بـ Ollama: {e}")
        return "حدث خطأ في الاتصال بالنموذج. يرجى المحاولة مجدداً."


# =========================
# CHAT LOOP
# =========================
print("=" * 50)
print("🤖 مرحبًا! أنا مساعدك في وزارة التجارة وتنمية الصادرات 🤖")
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

    if is_greeting(query):
        print(f"\n🤖 المساعد:\n{GREETING_RESPONSE}\n")
        continue

    if is_nonsense(query):
        print(f"\n🤖 المساعد:\n{NO_INFO_RESPONSE}\n")
        continue

    results = search(query)

    if not results:
        print(f"\n🤖 المساعد:\n{NO_INFO_RESPONSE}\n")
        continue

    context = build_context(results)
    print("\n🤖 المساعد:")
    answer = ask_llama(context, query)
    print(answer)
    print()