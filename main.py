import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# ==================================
# LOAD ALL TABLES
# ==================================

excel_path = r"C:\Users\assma\OneDrive\Bureau\chatbot_project\data.xlsx"

print("Reading Excel file...")

excel = pd.ExcelFile(excel_path)

all_tables = []

# read ALL sheets automatically
for sheet in excel.sheet_names:
    print("Reading table:", sheet)

    df = pd.read_excel(excel, sheet_name=sheet)

    df = df.fillna("").astype(str)
    df["source_table"] = sheet

    all_tables.append(df)

# merge everything
combined_df = pd.concat(all_tables, ignore_index=True)

print("Total rows:", len(combined_df))


# ==================================
#  CONVERT ROWS TO TEXT
# ==================================

documents = []

for _, row in combined_df.iterrows():
    text = " | ".join(map(str, row.values))
    documents.append(text)

print("Example document:")
print(documents[0])


# ==================================
#  CREATE EMBEDDINGS
# ==================================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ==================================


with open("embeddings.pkl", "wb") as f:
    pickle.dump(
        {
            "dataframe": combined_df,
            "documents": documents,
            "embeddings": embeddings
        },
        f
    )

print("✅ ALL TABLES EMBEDDED SUCCESSFULLY!")















