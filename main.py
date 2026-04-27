import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# ==================================
# LOAD ALL TABLES
# ==================================

excel_path = "data.xlsx"
excel = pd.ExcelFile(excel_path)  # FIX: was missing this line

all_documents = []  # list of dicts with text + metadata

for sheet in excel.sheet_names:
    print("Reading table:", sheet)
    df = pd.read_excel(excel, sheet_name=sheet, header=0)  # row 0 = headers
    df = df.fillna("").astype(str)

    # Get column names (these are the "questions"/field labels)
    columns = df.columns.tolist()

    # For each row (which is one record/answer set), build a rich text document
    for _, row in df.iterrows():
        parts = []
        for col in columns:
            val = str(row[col]).strip()
            if val and val not in ("nan", ""):
                parts.append(f"{col}: {val}")

        if parts:
            text = " | ".join(parts)
            all_documents.append({
                "text": text,
                "source_table": sheet,
                "procedure": str(row.get("التسمية_القانونية_بالعربية", "")).strip()
            })

print(f"Total documents: {len(all_documents)}")
if all_documents:
    print("Example document:")
    print(all_documents[0]["text"])

# ==================================
# CREATE EMBEDDINGS
# ==================================

print("Loading embedding model...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

texts = [doc["text"] for doc in all_documents]

print("Creating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

# ==================================
# SAVE
# ==================================

with open("embeddings.pkl", "wb") as f:
    pickle.dump(
        {
            "documents": all_documents,   # list of dicts (text + metadata)
            "texts": texts,               # plain text list for easy access
            "embeddings": embeddings
        },
        f
    )

print("✅ ALL TABLES EMBEDDED SUCCESSFULLY!")
print(f"📊 Total embedded documents: {len(all_documents)}")















