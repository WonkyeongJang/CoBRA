#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from tqdm import tqdm  # 프로그레스바 추가

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class CoBRA(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=64, output_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, L, input_dim]
        features = self.mlp(x)              # [B, L, hidden_dim]
        logits = self.classifier(features)  # [B, L, output_dim]
        return logits, features


def predict_from_pt(pt_path: str, model: nn.Module, device: torch.device):
    data = torch.load(pt_path, map_location=device)
    emb = data.get("embedding")
    seq = data.get("sequence", "")
    if emb is None or not isinstance(seq, str):
        raise KeyError(f"Required keys not found in {pt_path}")

    x = emb.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)

    preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
    valid_len = len(seq)
    return preds[:valid_len], seq
    

def run_folder(
    emb_dir: str,
    csv_path: str,
    device: torch.device,
):
    # ── 모델 로드 ────────────────────────────────────────
    model = CoBRA().to(device)
    weight_path = os.path.join(BASE_DIR, "weight", "CoBRA.pt")

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # .pt 파일 목록 준비
    pt_files = [f for f in sorted(os.listdir(emb_dir)) if f.endswith(".pt")]
    total = len(pt_files)

    # ── CSV 작성 및 예측 ─────────────────────────────────
    with open(csv_path, mode="w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["ID", "SEQUENCE", "BINDING_SITE"] )  # 헤더

        # tqdm으로 진행률 표시
        for fname in tqdm(pt_files, total=total, desc="Predicting", unit="file"):
            seq_id = os.path.splitext(fname)[0]
            pt_path = os.path.join(emb_dir, fname)
            try:
                preds, seq = predict_from_pt(pt_path, model, device)
                binary_str = "".join(map(str, preds.tolist()))
                writer.writerow([seq_id, seq, binary_str])
            except Exception as e:
                tqdm.write(f"Error on {seq_id}: {e}")

    print(f"The results have been saved to '{csv_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="CoBRA prediction: binary label prediction from .pt embedding file"
    )
    parser.add_argument(
        "emb_dir",
        help="Directory where embedding (.pt) files"
    )
    parser.add_argument(
        "csv_path",
        help="CSV file to save results (.csv only)"
    )
    parser.add_argument(
        "device", nargs="?", default=None,
        help="Compute device (cuda or cpu). Auto-detected if not specified"
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_folder(args.emb_dir, args.csv_path, device)

if __name__ == "__main__":
    main()