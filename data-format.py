import re
import pandas as pd

# 1. Load your original file
df = pd.read_csv("data/definitions-org.csv")

# Helper: split a cell containing "1. ... 2. ... 3. ..."
def split_definitions_cell(text: str):
    if not isinstance(text, str) or not text.strip():
        return []

    # Look for lines that start with:  number + dot + space  (e.g., "1. ", "2. ")
    pattern = re.compile(r'(?m)^\s*(\d+)\.\s+')
    matches = list(pattern.finditer(text))

    # If there are no numbered markers, treat the whole thing as a single definition
    if not matches:
        return [(None, text.strip())]

    parts = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        num = int(m.group(1))
        chunk = text[start:end].strip()
        if chunk:
            parts.append((num, chunk))

    return parts

# 2. Build expanded rows
expanded_rows = []

for _, row in df.iterrows():
    defs = split_definitions_cell(row["definition"])

    # If for some reason nothing comes back, still preserve the row
    if not defs:
        expanded_rows.append({
            "title": row["title"],
            "year": row["year"],
            "definition_number": None,
            "definition": None,
            "Cluster": row.get("Cluster", None),
            "Notes": row.get("Notes", None),
        })
    else:
        for num, dtext in defs:
            expanded_rows.append({
                "title": row["title"],
                "year": row["year"],
                "definition_number": num,               # 1, 2, 3, ...
                "definition": dtext,                    # cleaned single definition text
            })

expanded = pd.DataFrame(expanded_rows)

# 3. Save it out
expanded.to_csv("data/definitions_expanded.csv", index=False)

print(f"Original rows: {len(df)}")
print(f"Expanded rows (one per definition): {len(expanded)}")
