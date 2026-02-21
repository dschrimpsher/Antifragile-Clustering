import unicodedata
import re
import pandas as pd
from common import *

def normalize_text(s):
    if pd.isna(s):
        return ""

    s = str(s)

    # Normalize Unicode (fix weird encodings)
    s = unicodedata.normalize("NFKD", s)

    # Remove control characters
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", s)

    # Remove smart quotes and weird punctuation
    s = s.replace("“", "").replace("”", "")
    s = s.replace("‘", "").replace("’", "")
    s = s.replace("â€™", "")
    s = s.replace("€", "")
    s = s.replace("â", "")
    s = s.replace("œ", "")
    s = s.replace("", "")


    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    return s.strip().lower()

# 1. Load your original file
df = pd.read_csv("data/Copy of definition assignments.xlsx - Sheet1.csv")
df2 = pd.read_csv("data/definitions_encoded_with_uuid.csv")

for col in ANNS_COLUMNS:
    if col in df.columns:
        df[col] = df[col].apply(normalize_text)

# Normalize mechanism columns: NaN -> 0, enforce int8
for col in MECH_COLS:
    if col in df.columns:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
              .fillna(0)
              .astype("int8")
        )


for col in ENCODED_COLS:
    if col in df2.columns:
        df2[col] = df2[col].apply(normalize_text)

# Normalize mechanism columns: NaN -> 0, enforce int8
for col in MECH_COLS:
    if col in df2.columns:
        df2[col] = (
            pd.to_numeric(df2[col], errors="coerce")
              .fillna(0)
              .astype("int8")
        )



df.to_csv("data/ann_normalized.csv", index=False)
df2.to_csv("data/encoded_uuid_normalized.csv", index=False)
