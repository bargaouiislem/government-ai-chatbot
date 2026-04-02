import pandas as pd

# Open Excel file
excel = pd.ExcelFile(r"C:\Users\assma\OneDrive\Bureau\chatbot_project\data.xlsx")

all_tables = []

# Read ALL sheets automatically
for sheet in excel.sheet_names:
    print("Reading table:", sheet)

    df = pd.read_excel(excel, sheet_name=sheet)

    # convert everything to text (important for AI)
    df = df.fillna("").astype(str)

    # keep sheet name (so AI knows source)
    df["source_table"] = sheet

    all_tables.append(df)

# Merge ALL tables together
combined_df = pd.concat(all_tables, ignore_index=True)

print("✅ Total rows loaded:", len(combined_df))