from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
from ollama import Client

# =========================
# GLOBALS
# =========================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = Client(host=OLLAMA_HOST)

model = None
all_docs = []
texts = []
embeddings = None

# =========================
# CONSTANTS
# =========================
# Known Arabic administrative keywords that are always valid
ARABIC_PROCEDURE_KEYWORDS = [
    "شهادة", "رخصة", "وثيقة", "وثائق", "مستند", "ملف", "طلب", "إجراء",
    "تصدير", "استيراد", "تجارة", "وزارة", "إدارة", "خدمة", "تسجيل",
    "شروط", "مراحل", "تكلفة", "معلوم", "صلاحية", "متابعة", "قانون",
    "أوراق", "مطلوب", "ضروري", "هياكل", "جهات", "جهوية", "مواد", "خطرة"
]

NO_INFO_RESPONSE = "عذراً، لا توجد معلومات كافية حول هذا الموضوع."
GREETING_RESPONSE = "أهلاً وسهلاً! 😊 كيف يمكنني مساعدتك اليوم؟ يمكنك سؤالي عن أي إجراء أو خدمة تقدمها وزارة التجارة وتنمية الصادرات."

GREETING_PATTERNS = [
    r"^(أهلا|اهلا|مرحبا|مرحباً|السلام عليكم|سلام|صباح الخير|مساء الخير|هلا|يسلمو|شكراً|شكرا|وداعا|مع السلامة|إلى اللقاء)\b",
    r"^(hi|hello|hey|bonjour|salut|bonsoir|bye|goodbye|thanks|thank you|merci|ok|okay|yes|no|yep|nope|lol|haha)\b",
]

# =========================
# VALIDATION FUNCTIONS
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
    """
    Returns True only if the text looks like a real Arabic administrative question.
    Rejects:
    - Pure Latin keyboard spam (ojpo, snstsst, etc.)
    - Random character sequences with no Arabic letters
    - Very short inputs
    """
    t = text.strip()

    # Too short
    if len(t) < 4:
        return False

    # Count character types
    arabic_chars = sum(1 for c in t if '\u0600' <= c <= '\u06FF')
    latin_alpha  = sum(1 for c in t if c.isalpha() and c.isascii())
    total_alpha  = arabic_chars + latin_alpha

    # If no recognisable letters at all → reject
    if total_alpha == 0:
        return False

    # If it has Arabic letters → likely valid
    if arabic_chars >= 3:
        return True

    # If it's purely Latin letters (like ojpo, snstsst):
    # Only accept if it matches a known greeting/language word
    # Otherwise reject — it's random Latin keyboard spam
    if latin_alpha > 0 and arabic_chars == 0:
        # Check if it contains any known valid words
        known_latin = r"\b(what|how|where|when|who|is|are|the|a|an|document|procedure|fee|cost|step|condition|follow|legal)\b"
        if re.search(known_latin, t, re.IGNORECASE):
            return True
        # Otherwise it's garbage like "ojpo" or "snstsst"
        return False

    # Mixed but very low Arabic ratio → reject
    if total_alpha > 0 and arabic_chars / total_alpha < 0.3:
        return False

    return True


# =========================
# STARTUP
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, all_docs, texts, embeddings

    print("🔄 Loading embedding model...")
    try:
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("✅ Embedding model loaded!")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")

    pkl_path = "/app/embeddings.pkl"
    if os.path.exists(pkl_path):
        print("📂 Loading embeddings data...")
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if "documents" in data and isinstance(data["documents"][0], dict):
                all_docs = data["documents"]
                texts = data["texts"]
            else:
                texts = data["documents"]
                all_docs = [{"text": t, "source_table": "", "procedure": ""} for t in texts]
            embeddings = data["embeddings"]
            print(f"✅ Loaded {len(texts)} documents.")
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
    else:
        print("⚠️ embeddings.pkl not found! Run: docker exec -it tunisian-chatbot python main.py")

    print("🚀 API is ready!")
    yield
    print("👋 Shutting down...")


# =========================
# INIT APP
# =========================
app = FastAPI(title="وزارة التجارة - Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/app"), name="static")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []


# =========================
# SEARCH
# =========================
def search(query: str, top_k: int = 10, threshold: float = 0.25):
    if model is None or embeddings is None:
        return []

    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    best_score = float(similarities[top_indices[0]])
    print(f"🔍 Best similarity score: {best_score:.3f}")

    # Hard gate: best score must be above 0.40 to answer
    # This is the key fix — raised from 0.30 to 0.40
    if best_score < 0.40:
        print(f"⛔ Score {best_score:.3f} below gate 0.40 → no answer")
        return []

    filtered = [(i, float(similarities[i])) for i in top_indices if similarities[i] > threshold]
    return filtered if filtered else []


# =========================
# BUILD CONTEXT
# =========================
def build_context(filtered_results):
    seen_procedures = set()
    parts = []
    sources = []

    for idx, score in filtered_results:
        doc = all_docs[idx]
        proc = doc.get("procedure", "")
        section = doc.get("section", "")
        text = doc.get("text", texts[idx])
        if section == "كل المعلومات" and proc not in seen_procedures:
            seen_procedures.add(proc)
            parts.append(f"=== معلومات كاملة عن إجراء: {proc} ===\n{text}")
            if proc not in sources:
                sources.append(proc)

    for idx, score in filtered_results:
        doc = all_docs[idx]
        proc = doc.get("procedure", "")
        section = doc.get("section", "")
        text = doc.get("text", texts[idx])
        if section not in ("كل المعلومات", "سؤال وجواب") and proc not in seen_procedures:
            seen_procedures.add(proc)
            parts.append(f"=== {proc} - {section} ===\n{text}")
            if proc not in sources:
                sources.append(proc)

    if not parts:
        for idx, score in filtered_results:
            doc = all_docs[idx]
            proc = doc.get("procedure", "")
            section = doc.get("section", "")
            text = doc.get("text", texts[idx])
            if section == "سؤال وجواب" and proc not in seen_procedures:
                seen_procedures.add(proc)
                parts.append(f"=== {proc} ===\n{text}")
                if proc not in sources:
                    sources.append(proc)

    if not parts and filtered_results:
        idx, score = filtered_results[0]
        doc = all_docs[idx]
        parts.append(doc.get("text", texts[idx]))
        sources.append(doc.get("procedure", ""))

    return "\n\n---\n\n".join(parts), sources


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
def ask_llama(context: str, question: str) -> str:
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
        print(f"⚠️ Ollama error: {e}")
        return "حدث خطأ في الاتصال بالنموذج. يرجى المحاولة مجدداً."


# =========================
# ROUTES
# =========================
@app.get("/")
def serve_frontend():
    return FileResponse("/app/index1.html")


@app.get("/logo.png")
def serve_logo():
    return FileResponse("/app/logo.png", media_type="image/png")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_loaded": len(texts),
        "model_loaded": model is not None,
        "embeddings_loaded": embeddings is not None
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()

    if not query:
        return ChatResponse(response="الرجاء كتابة سؤال.")

    # Step 1: greeting detection
    if is_greeting(query):
        return ChatResponse(response=GREETING_RESPONSE, sources=[])

    # Step 2: validate the input is a real question
    # This catches Latin keyboard spam like "ojpo", "snstsst"
    if not is_valid_question(query):
        return ChatResponse(response=NO_INFO_RESPONSE, sources=[])

    if model is None or embeddings is None:
        return ChatResponse(
            response="النظام لم يتحمل بعد. يرجى الانتظار قليلاً والمحاولة مجدداً.",
            sources=[]
        )

    # Step 3: semantic search with hard 0.40 gate
    results = search(query)

    if not results:
        return ChatResponse(response=NO_INFO_RESPONSE, sources=[])

    context, sources = build_context(results)
    answer = ask_llama(context, query)

    return ChatResponse(response=answer, sources=[])