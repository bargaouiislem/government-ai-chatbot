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

# -------------------------------------------------------
# SYNONYM MAP: many ways to ask about the same section
# This is the KEY FIX for the synonym/paraphrase problem.
# We generate many QA pairs per section using different
# Arabic phrasings, so the embedding model can match
# user queries regardless of which words they use.
# -------------------------------------------------------
SYNONYM_QUESTIONS = {
    "الوثائق المطلوبة": [
        "ما هي الوثائق المطلوبة لـ {proc}؟",
        "ما هي الأوراق المطلوبة لـ {proc}؟",
        "ما هي الملفات الضرورية لـ {proc}؟",
        "ما هي المستندات اللازمة لـ {proc}؟",
        "ما هي الأوراق اللي نحتاجها لـ {proc}؟",
        "شنوا الوثائق المحتاجة لـ {proc}؟",
        "أش من وثائق نجيب لـ {proc}؟",
        "ما هي الأوراق الرسمية اللازمة لـ {proc}؟",
        "ما هي المستندات المطلوبة للحصول على {proc}؟",
        "ما هي الوثائق الإدارية الضرورية لـ {proc}؟",
        "ما هي الأوراق التي يجب تقديمها لـ {proc}؟",
        "ماذا أحضر معي لـ {proc}؟",
        "ما الذي أحتاجه من وثائق لـ {proc}؟",
        "قائمة الوثائق اللازمة لـ {proc}",
        "شنوا الوراق المحتاجة باش نعمل {proc}؟",
    ],
    "الشروط الضرورية": [
        "ما هي شروط {proc}؟",
        "ما هي الشروط الضرورية لـ {proc}؟",
        "ما هي متطلبات {proc}؟",
        "ما هي المتطلبات اللازمة لـ {proc}؟",
        "ما هي الشروط المطلوبة للحصول على {proc}؟",
        "ما الشروط الواجب توفرها لـ {proc}؟",
        "من يحق له التقدم لـ {proc}؟",
        "من يستطيع التقديم على {proc}؟",
        "ما شروط الاستفادة من {proc}؟",
        "ما هي الاشتراطات اللازمة لـ {proc}؟",
    ],
    "مراحل إنجاز الإجراء": [
        "ما هي مراحل {proc}؟",
        "ما هي خطوات {proc}؟",
        "كيف أقوم بـ {proc}؟",
        "كيف يتم {proc}؟",
        "ما هي إجراءات {proc}؟",
        "ما هي الإجراءات اللازمة لـ {proc}؟",
        "كيفاش نعمل {proc}؟",
        "شنوا خطوات {proc}؟",
        "كيف أنجز {proc}؟",
        "ما هي مسيرة إنجاز {proc}؟",
        "كيف تسير عملية {proc}؟",
        "ما هو مسار {proc}؟",
        "اشرح لي كيفية التقديم على {proc}",
    ],
    "معلوم الإجراء": [
        "كم تكلفة {proc}؟",
        "ما هو معلوم {proc}؟",
        "ما هو ثمن {proc}؟",
        "كم يكلف {proc}؟",
        "ما هو سعر {proc}؟",
        "ما هو المبلغ المطلوب لـ {proc}؟",
        "هل {proc} مجاني؟",
        "كم أدفع لـ {proc}؟",
        "ما هو الأداء المطلوب لـ {proc}؟",
        "ما هي رسوم {proc}؟",
        "كم يساوي {proc}؟",
        "شقدر تكلفة {proc}؟",
    ],
    "المعلوم الثابت": [
        "كم تكلفة {proc}؟",
        "ما هو المعلوم الثابت لـ {proc}؟",
        "ما هو المبلغ الثابت لـ {proc}؟",
        "ما هي الرسوم الثابتة لـ {proc}؟",
        "كم يكلف {proc}؟",
        "ما هو سعر {proc}؟",
        "ما هي تعريفة {proc}؟",
    ],
    "الإجراء بدون معلوم": [
        "هل {proc} مجاني؟",
        "هل {proc} بدون رسوم؟",
        "هل هناك مصاريف لـ {proc}؟",
        "كم تكلفة {proc}؟",
        "ما هي رسوم {proc}؟",
    ],
    "الجهات المتعهدة بقبول وإسداء الخدمة": [
        "أين أقدم طلب {proc}؟",
        "أين أذهب لـ {proc}؟",
        "في أي مكتب أتقدم لـ {proc}؟",
        "ما هي الجهة المسؤولة عن {proc}؟",
        "أين أودع ملف {proc}؟",
        "أين يمكنني طلب {proc}؟",
        "ما هو المكان الذي يمكنني فيه تقديم {proc}؟",
        "ما هي الإدارة المسؤولة عن {proc}؟",
        "وين نروح باش نطلب {proc}؟",
        "أين أتوجه لإنجاز {proc}؟",
        "ما هي الجهات التي تتولى {proc}؟",
    ],
    "الهياكل المشرفة على الإنشاء": [
        "من يشرف على {proc}؟",
        "ما هي الهيئة المسؤولة عن {proc}؟",
        "ما هي الجهة المشرفة على {proc}؟",
        "أين أقدم طلب {proc}؟",
        "ما هي الإدارة التي تتولى {proc}؟",
        "أين أتوجه لـ {proc}؟",
        "من يتولى {proc}؟",
    ],
    "آجال إنجاز الإجراء": [
        "ما هو أجل إنجاز {proc}؟",
        "كم يستغرق {proc}؟",
        "ما هي مدة {proc}؟",
        "كم من الوقت يحتاج {proc}؟",
        "في كم يوم يتم {proc}؟",
        "ما هو الوقت اللازم لـ {proc}؟",
        "ما هو الأجل القانوني لـ {proc}؟",
        "كم تستغرق معالجة {proc}؟",
        "قداش يخذ {proc}؟",
    ],
    "مدة صلاحية الإجراء": [
        "ما هي مدة صلاحية {proc}؟",
        "ما هي صلاحية {proc}؟",
        "كم تدوم صلاحية {proc}؟",
        "متى تنتهي صلاحية {proc}؟",
        "إلى متى يبقى {proc} صالحاً؟",
        "ما هي مدة سريان {proc}؟",
    ],
    "طريقة المتابعة": [
        "كيف أتابع ملف {proc}؟",
        "كيف أعرف حالة طلب {proc}؟",
        "كيف أتابع طلب {proc}؟",
        "كيفاش نتابع {proc}؟",
        "ما هي طريقة متابعة {proc}؟",
        "كيف يمكنني معرفة مآل {proc}؟",
        "كيف أعرف إذا تمت الموافقة على {proc}؟",
    ],
    "المراجع القانونية": [
        "ما هو الأساس القانوني لـ {proc}؟",
        "ما هي القوانين المنظمة لـ {proc}؟",
        "ما هي النصوص القانونية المتعلقة بـ {proc}؟",
        "ما هو الإطار القانوني لـ {proc}؟",
        "ما هي المراجع القانونية لـ {proc}؟",
        "ما هو المرجع التشريعي لـ {proc}؟",
    ],
}


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

    # Document 3: Rich Synonym QA pairs
    # For every section that has data, we generate ALL synonym questions
    # so that no matter how the user phrases their query, the embedding
    # model can find the matching answer.
    for section_label, question_templates in SYNONYM_QUESTIONS.items():
        answer = sections.get(section_label, "")
        # Also check fallback synonyms for cost (معلوم)
        if not answer and section_label == "معلوم الإجراء":
            answer = sections.get("المعلوم الثابت", "") or sections.get("الإجراء بدون معلوم", "")
        if not answer and section_label == "الجهات المتعهدة بقبول وإسداء الخدمة":
            answer = sections.get("الهياكل المشرفة على الإنشاء", "")

        if answer and answer.strip():
            for question_template in question_templates:
                question = question_template.format(proc=proc_name)
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













