import pandas as pd
import os

# Path ke folder dataset NusaX-MT
base_path = "datasets/mt"

# Path folder output hasil preprocessing
out_path = "data/processed"
os.makedirs(out_path, exist_ok=True)

# Split yang mau diproses
splits = ["train", "valid", "test"]

for split in splits:
    print(f"Processing {split}...")
    file_path = os.path.join(base_path, f"{split}.csv")

    # Baca file CSV
    df = pd.read_csv(file_path)
    
    # Ambil kolom Indonesia dan Jawa
    if "indonesian" not in df.columns or "javanese" not in df.columns:
        raise ValueError("Kolom 'indonesian' atau 'javanese' tidak ditemukan!")
    
    src_texts = df["indonesian"].astype(str).str.strip()
    tgt_texts = df["javanese"].astype(str).str.strip()
    
    # Simpan hasil ke file teks
    src_out = os.path.join(out_path, f"{split}.id")
    tgt_out = os.path.join(out_path, f"{split}.jv")
    
    src_texts.to_csv(src_out, index=False, header=False)
    tgt_texts.to_csv(tgt_out, index=False, header=False)
    
    print(f"✔ {split} selesai, {len(df)} baris disimpan.")

print("✅ Semua selesai! File hasil ada di folder data/processed/")
