import torch
import matplotlib.pyplot as plt
import seaborn as sns
from models.seq2seq_attention import Seq2SeqAttn, Encoder, Decoder, Attention
import sentencepiece as spm
import numpy as np

# ---- Load SentencePiece ----
sp_src = spm.SentencePieceProcessor()
sp_tgt = spm.SentencePieceProcessor()
sp_src.load("spm_src.model")
sp_tgt.load("spm_tgt.model")

# ---- Load Model ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# buat komponen model
INPUT_DIM = len(sp_src)
OUTPUT_DIM = len(sp_tgt)
EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
DROPOUT = 0.2

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT, attn)
model = Seq2SeqAttn(enc, dec, DEVICE).to(DEVICE)

# load bobot
state_dict = torch.load("model_attention.pt", map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ---- Input contoh ----
sentence = "Iya benar, dia sedang jaga warung."
tokens = [sp_src.bos_id()] + sp_src.encode(sentence) + [sp_src.eos_id()]
src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(DEVICE)

# ---- Proses encoding & decoding ----
with torch.no_grad():
    encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # === MODE 1: pakai kalimat referensi (teacher forcing) ===
    use_reference = True  # ganti ke False kalau mau lihat hasil prediksi model
    attentions = []
    decoded_tokens = []

    if use_reference:
        tgt_sentence = "Iya bener, deknen lagi jaga warung."
        tgt_tokens = [sp_tgt.bos_id()] + sp_tgt.encode(tgt_sentence) + [sp_tgt.eos_id()]
        tgt_tensor = torch.LongTensor(tgt_tokens).to(DEVICE)

        for t in range(1, len(tgt_tokens)):
            input_token = tgt_tensor[t-1].unsqueeze(0)
            output, hidden, cell, attn = model.decoder(input_token, hidden, cell, encoder_outputs)
            attentions.append(attn.squeeze(0).cpu())
            decoded_tokens.append(tgt_tensor[t].item())

    else:
        # === MODE 2: prediksi bebas (tanpa referensi) ===
        input_token = torch.LongTensor([sp_tgt.bos_id()]).to(DEVICE)
        for _ in range(10):
            output, hidden, cell, attn = model.decoder(input_token, hidden, cell, encoder_outputs)
            attentions.append(attn.squeeze(0).cpu())
            top1 = output.argmax(1)
            decoded_tokens.append(top1.item())
            if top1.item() == sp_tgt.eos_id():
                break
            input_token = top1

# ---- Visualisasi attention ----
src_tokens = ["<s>"] + sp_src.decode(tokens[1:-1]).split() + ["</s>"]
tgt_tokens = ["<s>"] + sp_tgt.decode(decoded_tokens).split()

att_matrix = torch.cat([torch.tensor(a) for a in attentions], dim=0).squeeze().cpu().numpy()

plt.figure(figsize=(8, 6))
sns.heatmap(att_matrix, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap="YlGnBu")
plt.xlabel("Source (Input Sentence)")
plt.ylabel("Target (Predicted Translation)")
plt.title("Attention Alignment Heatmap")
plt.show()
