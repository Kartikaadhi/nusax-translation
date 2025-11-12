import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import sentencepiece as spm
from models.seq2seq import Encoder, Decoder, Seq2Seq
from models.seq2seq_attention import Encoder as AttnEncoder, Decoder as AttnDecoder, Attention, Seq2SeqAttn

# ==== CONFIG ====
SRC_PATH = "data/processed/train.id"
TGT_PATH = "data/processed/train.jv"
MODEL_TYPE = "attention"  # ganti ke "seq2seq" kalau mau model tanpa attention
EPOCHS = 5
BATCH_SIZE = 16
LR = 0.001
EMB_DIM = 128
HID_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT = 0.3
TEACHER_FORCING = 0.7

# ==== DATASET ====
class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines):
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
    def __len__(self):
        return len(self.src_lines)
    def __getitem__(self, idx):
        return self.src_lines[idx], self.tgt_lines[idx]

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# ==== LOAD DATA ====
src_lines = read_lines(SRC_PATH)
tgt_lines = read_lines(TGT_PATH)

assert len(src_lines) == len(tgt_lines), "Source dan target tidak sama panjang!"

# ==== TOKENIZER (SentencePiece) ====
if not os.path.exists("spm_src.model"):
    with open("spm_src.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(src_lines))
    spm.SentencePieceTrainer.Train("--input=spm_src.txt --model_prefix=spm_src --vocab_size=2000 --character_coverage=0.9995")

if not os.path.exists("spm_tgt.model"):
    with open("spm_tgt.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tgt_lines))
    spm.SentencePieceTrainer.Train("--input=spm_tgt.txt --model_prefix=spm_tgt --vocab_size=2000 --character_coverage=0.9995")

sp_src = spm.SentencePieceProcessor(model_file="spm_src.model")
sp_tgt = spm.SentencePieceProcessor(model_file="spm_tgt.model")

def encode_pair(src, tgt):
    return [sp_src.bos_id()] + sp_src.encode(src, out_type=int) + [sp_src.eos_id()], \
           [sp_tgt.bos_id()] + sp_tgt.encode(tgt, out_type=int) + [sp_tgt.eos_id()]

pairs = [encode_pair(s, t) for s, t in zip(src_lines, tgt_lines)]

# ==== PAD + DATALOADER ====
def pad_sequences(sequences, pad_token=0):
    max_len = max(len(s) for s in sequences)
    return [s + [pad_token]*(max_len - len(s)) for s in sequences]

class Collate:
    def __call__(self, batch):
        src, tgt = zip(*batch)
        src = pad_sequences(src)
        tgt = pad_sequences(tgt)
        src = torch.tensor(src, dtype=torch.long).transpose(0,1)
        tgt = torch.tensor(tgt, dtype=torch.long).transpose(0,1)
        return src, tgt

dataset = TranslationDataset([p[0] for p in pairs], [p[1] for p in pairs])
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collate())

# ==== MODEL ====
input_dim = len(sp_src)
output_dim = len(sp_tgt)

if MODEL_TYPE == "seq2seq":
    enc = Encoder(input_dim, EMB_DIM, HID_DIM)
    dec = Decoder(output_dim, EMB_DIM, HID_DIM)
    model = Seq2Seq(enc, dec, DEVICE)
else:
    attn = Attention(HID_DIM, HID_DIM)
    enc = AttnEncoder(input_dim, EMB_DIM, HID_DIM, HID_DIM, 0.2)
    dec = AttnDecoder(output_dim, EMB_DIM, HID_DIM, HID_DIM, 0.2, attn)
    model = Seq2SeqAttn(enc, dec, DEVICE)

model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ==== TRAIN LOOP ====
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING)
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        tgt = tgt[1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # ⬅️ penting
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(loader):.4f}")

torch.save(model.state_dict(), f"model_{MODEL_TYPE}.pt")
print("✅ Training selesai & model tersimpan!")
