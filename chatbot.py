import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = Client(host=OLLAMA_HOST)

# =========================
# CONSTANTS
# =========================
NO_INFO_RESPONSE = "عذراً، لا توجد معلومات كافية حول هذا الموضوع."
GREETING_RESPONSE = "أهلاً وسهلاً! 😊 كيف يمكنني مساعدتك اليوم؟ يمكنك سؤالي عن أي إجراء أو خدمة تقدمها وزارة التجارة وتنمية الصادرات."

GREETING_PATTERNS = [
    r"^(أهلا|اهلا|مرحبا|مرحباً|السلام عليكم|سلام|صباح الخير|مساء الخير|هلا|يسلمو|شكراً|شكرا|وداعا|مع السلامة|إلى اللقاء)\b",
    r"^(hi|hello|hey|bonjour|salut|bonsoir|bye|goodbye|thanks|thank you|merci|ok|okay|yes|no|yep|nope|lol|haha)\b",
]

# =========================
# VALIDATION
# =========================
def is_greeting(text: str) -> bool:
    t = text.strip()
    if len(t) < 3:
        return True
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE | re.UNICODE):
            return True
    return False


def is_valid_question(text: str) -> bool:
    t = text.strip()
    if len(t) < 4:
        return False

    arabic_chars = sum(1 for c in t if '\u0600' <= c <= '\u06FF')
    latin_alpha  = sum(1 for c in t if c.isalpha() and c.isascii())
    total_alpha  = arabic_chars + latin_alpha

    if total_alpha == 0:
        return False
    if arabic_chars >= 3:
        return True

    if latin_alpha > 0 and arabic_chars == 0:
        known_latin = r"\b(what|how|where|when|who|is|are|document|procedure|fee|cost|step|condition|follow|legal)\b"
        if re.search(known_latin, t, re.IGNORECASE):
            return True
        return False

    if total_alpha > 0 and arabic_chars / total_alpha < 0.3:
        return False

    return True


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
def search(query, top_k=10, threshold=0.25):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    best_score = float(similarities[top_indices[0]])
    print(f"🔍 Best score: {best_score:.3f}")

    # Hard gate at 0.40 — rejects all nonsense
    if best_score < 0.40:
        print(f"⛔ Below 0.40 gate → no answer")
        return []

    filtered = [(i, float(similarities[i])) for i in top_indices if similarities[i] > threshold]
    if filtered:
        print(f"📄 Top results:")
        for idx, score in filtered[:3]:
            proc = all_docs[idx].get("procedure", "")
            print(f"  [{score:.3f}] {proc[:50]}")
    return filtered if filtered else []


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
# FIX NUMBERING
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

قواعد صارمة لا استثناء فيها:
١. أجب بالعربية فقط — ممنوع أي كلمة بالفرنسية أو الإنجليزية أو أي لغة أخرى.
٢. استخدم فقط المعلومات الموجودة في السياق — لا تخترع أي معلومة.
٣. إذا لم تجد الإجابة قل فقط: "عذراً، لا توجد معلومات كافية حول هذا الموضوع."
٤. لا تذكر أسماء جداول أو قواعد بيانات أو مصادر.
٥. رقّم القوائم هكذا: 1. ثم 2. ثم 3. — ممنوع استخدام * أو - أو •
٦. أجب على هذا السؤال فقط، لا تضف معلومات أخرى.

=== المعلومات المتاحة ===
{context}

=== سؤال المواطن ===
{question}

=== الإجابة بالعربية فقط ==="""

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
            return NO_INFO_RESPONSE

        answer = fix_numbering(answer)
        return answer.strip()

    except Exception as e:
        print(f"⚠️ خطأ: {e}")
        return "حدث خطأ في الاتصال بالنموذج. يرجى المحاولة مجدداً."


# =========================
# CHAT LOOP
# =========================
print("=" * 50)
print("🤖 مرحبًا! أنا مساعدك في وزارة التجارة وتنمية الصادرات")
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

    if not is_valid_question(query):
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