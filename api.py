from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from ollama import Client

# =========================
# INIT APP
# =========================
app = FastAPI(title="وزارة التجارة - Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (logo.png, index1.html, etc.)
app.mount("/static", StaticFiles(directory="/app"), name="static")

# =========================
# LOAD OLLAMA CLIENT
# =========================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = Client(host=OLLAMA_HOST)

# =========================
# LOAD MODEL & DATA
# =========================
print("🔄 Loading embedding model...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("📂 Loading embeddings data...")
with open("/app/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

# Support both old and new embeddings.pkl format
if "documents" in data and isinstance(data["documents"][0], dict):
    all_docs = data["documents"]
    texts = data["texts"]
else:
    texts = data["documents"]
    all_docs = [{"text": t, "source_table": "", "procedure": ""} for t in texts]

embeddings = data["embeddings"]
print(f"✅ Loaded {len(texts)} documents.\n")


# =========================
# REQUEST / RESPONSE MODELS
# =========================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []


# =========================
# SEARCH
# =========================
def search(query: str, top_k: int = 5, threshold: float = 0.25):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    filtered = [(i, float(similarities[i])) for i in top_indices if similarities[i] > threshold]

    if not filtered:
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if best_score > 0.15:
            return [(best_idx, best_score)]
        return []

    return filtered


# =========================
# BUILD CONTEXT
# =========================
def build_context(filtered_results):
    parts = []
    sources = []
    for idx, score in filtered_results:
        doc = all_docs[idx]
        src = doc.get("source_table", "")
        text = doc.get("text", texts[idx])
        parts.append(f"[مصدر: {src}]\n{text}")
        if src and src not in sources:
            sources.append(src)
    return "\n\n---\n\n".join(parts), sources


# =========================
# ASK LLAMA
# =========================
def ask_llama(context: str, question: str) -> str:
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
            options={"temperature": 0.1, "num_predict": 512}
        )
        if isinstance(response, dict):
            return response["message"]["content"]
        else:
            return response.message.content
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return "حدث خطأ في الاتصال بالنموذج. يرجى المحاولة مجدداً."


# =========================
# ROUTES
# =========================
@app.get("/")
def serve_frontend():
    return FileResponse("/app/index1.html")

@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": len(texts)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()

    if not query:
        return ChatResponse(response="الرجاء كتابة سؤال.")

    results = search(query)

    if not results:
        return ChatResponse(
            response="لا توجد معلومات كافية في قاعدة البيانات حول هذا الموضوع.",
            sources=[]
        )

    context, sources = build_context(results)
    answer = ask_llama(context, query)

    return ChatResponse(response=answer, sources=sources)