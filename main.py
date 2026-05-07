import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# ==================================
# CONFIGURATION
# ==================================

excel_path = "data_cleaned.xlsx"

SHEET_LABELS = {
    "template_proc_struct_resp_creat": "الهياكل المشرفة على الإنشاء",
    "template_structure_region":       "الجهات المتعهدة بقبول وإسداء الخدمة",
    "template_proc_region":            "الجهات الجهوية المتعهدة",
    "template_type_docs":              "الوثائق المطلوبة",
    "temp-legislation":                "المراجع القانونية",
    "template_etape_traitement":       "مراحل إنجاز الإجراء",
    "temp-sans frais":                 "الإجراء بدون معلوم",
    "template_condition":              "الشروط الضرورية",
    "temp-validite":                   "مدة صلاحية الإجراء",
    "tempalte_delais_execution":       "آجال إنجاز الإجراء",
    "temp-frais":                      "معلوم الإجراء",
    "template_avec_frais_fixe":        "المعلوم الثابت",
    "template_preuve_paiement":        "وسيلة إثبات الدفع",
    "template_suivi_service":          "طريقة المتابعة",
}

# ==================================
# LOAD ALL SHEETS
# ==================================

excel = pd.ExcelFile(excel_path)
procedure_data = defaultdict(dict)  # { proc_name: { section_label: text } }

for sheet in excel.sheet_names:
    label = SHEET_LABELS.get(sheet, sheet)
    df = pd.read_excel(excel, sheet_name=sheet, header=0)
    df = df.fillna("")

    if df.empty or len(df.columns) == 0:
        continue

    # First column = procedure name
    proc_col = df.columns[0]

    for _, row in df.iterrows():
        proc_name = str(row[proc_col]).strip()
        if not proc_name or proc_name in ("nan", "None", ""):
            continue

        # Collect all values from other columns into one clean text
        parts = []
        for col in df.columns[1:]:
            val = str(row[col]).strip()
            if not val or val in ("nan", "None", ""):
                continue

            col_label = str(col).strip()

            # Clean up newlines inside cells → Arabic comma separated list
            if "\n" in val:
                items = [v.strip() for v in val.split("\n") if v.strip() and v.strip() not in ("nan", "None")]
                val = "، ".join(items)

            if val:
                parts.append(f"{col_label}: {val}")

        if parts:
            section_text = " | ".join(parts)
            if label in procedure_data[proc_name]:
                procedure_data[proc_name][label] += " | " + section_text
            else:
                procedure_data[proc_name][label] = section_text

print(f"✅ Loaded {len(procedure_data)} unique procedures:")
for p in procedure_data:
    print(f"  - {p}")

# ==================================
# BUILD DOCUMENTS
# ==================================

all_documents = []

for proc_name, sections in procedure_data.items():

    # Document 1: Full merged document
    full_parts = [f"الإجراء: {proc_name}"]
    for section_label, section_text in sections.items():
        full_parts.append(f"{section_label}: {section_text}")
    full_text = "\n".join(full_parts)
    all_documents.append({
        "text": full_text,
        "source_table": "معلومات_كاملة",
        "procedure": proc_name,
        "section": "كل المعلومات"
    })

    # Document 2+: One doc per section
    for section_label, section_text in sections.items():
        section_doc = f"الإجراء: {proc_name}\n{section_label}: {section_text}"
        all_documents.append({
            "text": section_doc,
            "source_table": section_label,
            "procedure": proc_name,
            "section": section_label
        })

    # Document 3: Synthetic QA in Arabic only
    qa_pairs = [
        (f"ما هي الوثائق المطلوبة لـ {proc_name}؟",
         sections.get("الوثائق المطلوبة", "")),
        (f"ما هي شروط {proc_name}؟",
         sections.get("الشروط الضرورية", "")),
        (f"ما هي مراحل {proc_name}؟",
         sections.get("مراحل إنجاز الإجراء", "")),
        (f"كم تكلفة {proc_name}؟",
         sections.get("معلوم الإجراء", "") or sections.get("المعلوم الثابت", "") or sections.get("الإجراء بدون معلوم", "")),
        (f"أين أقدم طلب {proc_name}؟",
         sections.get("الجهات المتعهدة بقبول وإسداء الخدمة", "") or sections.get("الهياكل المشرفة على الإنشاء", "")),
        (f"ما هو أجل إنجاز {proc_name}؟",
         sections.get("آجال إنجاز الإجراء", "")),
        (f"ما هي مدة صلاحية {proc_name}؟",
         sections.get("مدة صلاحية الإجراء", "")),
        (f"كيف أتابع ملف {proc_name}؟",
         sections.get("طريقة المتابعة", "")),
        (f"ما هو الأساس القانوني لـ {proc_name}؟",
         sections.get("المراجع القانونية", "")),
        (f"من يشرف على {proc_name}؟",
         sections.get("الهياكل المشرفة على الإنشاء", "")),
        (f"ما هي الجهات المتعهدة بقبول {proc_name}؟",
         sections.get("الجهات المتعهدة بقبول وإسداء الخدمة", "")),
    ]

    for question, answer in qa_pairs:
        if answer and answer.strip():
            qa_text = f"سؤال: {question}\nجواب: {answer}"
            all_documents.append({
                "text": qa_text,
                "source_table": "أسئلة_وأجوبة",
                "procedure": proc_name,
                "section": "سؤال وجواب"
            })

print(f"\n📊 Total documents: {len(all_documents)}")

# ==================================
# CREATE EMBEDDINGS
# ==================================

print("\n🔄 Loading embedding model...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

texts = [doc["text"] for doc in all_documents]

print(f"🧮 Creating embeddings for {len(texts)} documents...")
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
            "documents": all_documents,
            "texts": texts,
            "embeddings": embeddings,
            "procedure_data": dict(procedure_data),
        },
        f
    )

print("\n✅ EMBEDDINGS SAVED SUCCESSFULLY!")
print(f"📊 Total embedded documents: {len(all_documents)}")













