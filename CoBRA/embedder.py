#!/usr/bin/env python3
import os
import sys
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
RINALMO_PATH = os.path.join(PARENT_DIR, "RiNALMo")

sys.path.insert(0, str(RINALMO_PATH))

import torch
import torch.nn.functional as F
from Bio import SeqIO
from rinalmo.pretrained import get_pretrained_model

DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_LEN     = 161
VALID_BASES = {"A", "U", "G", "C"}

def load_model(device=DEVICE):
    model, alphabet = get_pretrained_model(model_name="giga-v1")
    return model.to(device).eval(), alphabet


def embed_fasta(fasta_path, output_dir, model, alphabet,
                device=DEVICE, max_len=MAX_LEN, valid_bases=VALID_BASES):
    os.makedirs(output_dir, exist_ok=True)

    # ← 한 번에 리스트로 읽기
    records = list(SeqIO.parse(fasta_path, "fasta"))
    total   = len(records)

    for rec in tqdm(records, total=total, desc="Embedding", unit="seq"):
        seq_id = rec.id
        seq = str(rec.seq).upper().replace("T", "U")

        if set(seq) - valid_bases:
            tqdm.write(f"[!] {seq_id}: invalid chars, skipped")
            continue
        if not (5 <= len(seq) <= max_len):
            tqdm.write(f"[!] {seq_id}: length out of range, skipped")
            continue

        token_ids = alphabet.batch_tokenize([seq])
        tokens    = torch.tensor(token_ids, dtype=torch.int64, device=device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            rep = model(tokens)["representation"].squeeze(0)

        token_emb = rep[1:1+len(seq)]
        if token_emb.size(0) != len(seq):
            tqdm.write(f"[!] {seq_id}: length mismatch, skipped")
            continue

        padded = F.pad(token_emb, (0,0,0, max_len - token_emb.size(0)))
        save_path = os.path.join(output_dir, f"{seq_id}.pt")
        torch.save({"sequence": seq, "embedding": padded.cpu()}, save_path)
        # tqdm.write(f"[+] {seq_id} ✔ Saved (shape {padded.shape})")
        
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Extract RiNALMo giga-v1 embeddings from a FASTA."
    )
    p.add_argument("fasta",  help="input FASTA file")
    p.add_argument("outdir", help="output directory for .pt files")
    args = p.parse_args()

    model, alphabet = load_model()
    embed_fasta(args.fasta, args.outdir, model, alphabet)
