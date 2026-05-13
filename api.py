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
def search(query: str, top_k: int = 10, threshold: float = 0.20):
    """
    Search for most relevant documents using cosine similarity.
    top_k is higher than before to improve recall for synonyms.
    """
    if model is None or embeddings is None:
        return []

    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    filtered = [(i, float(similarities[i])) for i in top_indices if similarities[i] > threshold]

    if not filtered:
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if best_score > 0.12:
            return [(best_idx, best_score)]
        return []

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
    sources = []

    # Priority 1: full procedure documents
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

    # Priority 2: section docs for procedures not yet covered
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

    # Priority 3: QA pairs as supplementary context if no full doc found
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

    # Fallback: just take the best result
    if not parts and filtered_results:
        idx, score = filtered_results[0]
        doc = all_docs[idx]
        parts.append(doc.get("text", texts[idx]))
        sources.append(doc.get("procedure", ""))

    return "\n\n---\n\n".join(parts), sources

# =========================
# ASK LLAMA
# =========================
def ask_llama(context: str, question: str) -> str:
    context = context[:6000]

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
                "num_predict": 1024,
            }
        )
        if isinstance(response, dict):
            answer = response["message"]["content"]
        else:
            answer = response.message.content

        # Safety: if the model still slips into non-Arabic, warn the user
        # (detect by checking if the answer contains Arabic characters at all)
        arabic_chars = sum(1 for c in answer if '\u0600' <= c <= '\u06FF')
        if arabic_chars < 10:
            return "حدث خطأ في معالجة الإجابة. يرجى إعادة صياغة سؤالك بالعربية والمحاولة مجدداً."

        return answer

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

    if model is None or embeddings is None:
        return ChatResponse(
            response="النظام لم يتحمل بعد. يرجى الانتظار قليلاً والمحاولة مجدداً.",
            sources=[]
        )

    results = search(query)

    if not results:
        return ChatResponse(
            response="لا توجد معلومات كافية في قاعدة البيانات حول هذا الموضوع.",
            sources=[]
        )

    context, sources = build_context(results)
    answer = ask_llama(context, query)

    return ChatResponse(response=answer, sources=[])