import pandas as pd
import os

# pastiin folder data ada
os.makedirs("data", exist_ok=True)

# baca file valid.csv
df = pd.read_csv("datasets/mt/valid.csv")

# ubah sesuai kolom yang bener
src_lang = "indonesian"
tgt_lang = "english"

# ambil kolom teks
src_texts = df[src_lang].astype(str).tolist()
tgt_texts = df[tgt_lang].astype(str).tolist()

# simpan ke file txt
with open("data/src_val.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(src_texts))

with open("data/tgt_val.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(tgt_texts))

print("✅ File src_val.txt dan tgt_val.txt berhasil dibuat!")
