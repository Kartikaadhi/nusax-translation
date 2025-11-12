import torch
import sentencepiece as spm
from sacrebleu import corpus_bleu
from models.seq2seq import Encoder, Decoder, Seq2Seq
from models.seq2seq_attention import Encoder as AttnEncoder, Decoder as AttnDecoder, Attention, Seq2SeqAttn
from rouge_score import rouge_scorer

# ==== CONFIG ====
SRC_PATH = "data/processed/test.id"
TGT_PATH = "data/processed/test.jv"
MODEL_TYPE = "attention"   # ganti ke "seq2seq" kalau mau model tanpa attention
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD TOKENIZER ====
sp_src = spm.SentencePieceProcessor(model_file="spm_src.model")
sp_tgt = spm.SentencePieceProcessor(model_file="spm_tgt.model")

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

src_lines = read_lines(SRC_PATH)
tgt_lines = read_lines(TGT_PATH)

# ==== PREPARE MODEL ====
input_dim = len(sp_src)
output_dim = len(sp_tgt)
emb_dim = 128
hid_dim = 256

if MODEL_TYPE == "seq2seq":
    enc = Encoder(input_dim, emb_dim, hid_dim)
    dec = Decoder(output_dim, emb_dim, hid_dim)
    model = Seq2Seq(enc, dec, DEVICE)
    model.load_state_dict(torch.load("model_seq2seq.pt", map_location=DEVICE))
else:
    attn = Attention(hid_dim, hid_dim)
    enc = AttnEncoder(input_dim, emb_dim, hid_dim, hid_dim, 0.2)
    dec = AttnDecoder(output_dim, emb_dim, hid_dim, hid_dim, 0.2, attn)
    model = Seq2SeqAttn(enc, dec, DEVICE)
    model.load_state_dict(torch.load("model_attention.pt", map_location=DEVICE))

model.eval()
model = model.to(DEVICE)

# ==== TRANSLATE FUNCTION ====
def translate_sentence(sentence):
    tokens = [sp_src.bos_id()] + sp_src.encode(sentence, out_type=int) + [sp_src.eos_id()]
    src_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        if MODEL_TYPE == "seq2seq":
            hidden, cell = model.encoder(src_tensor)
        else:
            encoder_outputs, hidden, cell = model.encoder(src_tensor)

    trg_indexes = [sp_tgt.bos_id()]
    for i in range(50):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(DEVICE)

        if MODEL_TYPE == "seq2seq":
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        else:
            output, hidden, cell, _ = model.decoder(trg_tensor, hidden, cell, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == sp_tgt.eos_id():
            break

    translated = sp_tgt.decode(trg_indexes[1:-1])
    return translated

# ==== EVALUATE BLEU ====
preds = []
for s in src_lines:
    preds.append(translate_sentence(s))

bleu = corpus_bleu(preds, [tgt_lines])
print(f"✅ Model: {MODEL_TYPE} | BLEU score: {bleu.score:.2f}")

# ==== EVALUATE ROUGE ====
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1, rouge2, rougeL = 0, 0, 0

for pred, ref in zip(preds, tgt_lines):
    scores = scorer.score(ref, pred)
    rouge1 += scores['rouge1'].fmeasure
    rouge2 += scores['rouge2'].fmeasure
    rougeL += scores['rougeL'].fmeasure

n = len(preds)
print(f"🧠 ROUGE-1: {rouge1/n:.4f} | ROUGE-2: {rouge2/n:.4f} | ROUGE-L: {rougeL/n:.4f}")
