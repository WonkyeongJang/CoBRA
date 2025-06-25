#!/usr/bin/env python3
"""
pipeline.py

CoBRA 전체 파이프라인: 1) FASTA → embedding 2) embedding → 예측
Usage:
    python pipeline.py <fasta> <emb_dir> <weight_path> [device]
"""
import argparse
import torch
import os
import sys

# ASCII banner font: doh.flf by Curtis Wanner (cwanner@acs.bu.edu), latest revision - 4/95
ASCII_BANNER = r"""
                                                                                                   
        CCCCCCCCCCCCC                 BBBBBBBBBBBBBBBBB   RRRRRRRRRRRRRRRRR                  AAA               
     CCC::::::::::::C                 B::::::::::::::::B  R::::::::::::::::R                A:::A              
   CC:::::::::::::::C                 B::::::BBBBBB:::::B R::::::RRRRRR:::::R              A:::::A             
  C:::::CCCCCCCC::::C                 BB:::::B     B:::::BRR:::::R     R:::::R            A:::::::A            
 C:::::C       CCCCCC   ooooooooooo     B::::B     B:::::B  R::::R     R:::::R           A:::::::::A           
C:::::C               oo:::::::::::oo   B::::B     B:::::B  R::::R     R:::::R          A:::::A:::::A          
C:::::C              o:::::::::::::::o  B::::BBBBBB:::::B   R::::RRRRRR:::::R          A:::::A A:::::A         
C:::::C              o:::::ooooo:::::o  B:::::::::::::BB    R:::::::::::::RR          A:::::A   A:::::A        
C:::::C              o::::o     o::::o  B::::BBBBBB:::::B   R::::RRRRRR:::::R        A:::::A     A:::::A       
C:::::C              o::::o     o::::o  B::::B     B:::::B  R::::R     R:::::R      A:::::AAAAAAAAA:::::A      
C:::::C              o::::o     o::::o  B::::B     B:::::B  R::::R     R:::::R     A:::::::::::::::::::::A     
 C:::::C       CCCCCCo::::o     o::::o  B::::B     B:::::B  R::::R     R:::::R    A:::::AAAAAAAAAAAAA:::::A    
  C:::::CCCCCCCC::::Co:::::ooooo:::::oBB:::::BBBBBB::::::BRR:::::R     R:::::R   A:::::A             A:::::A   
   CC:::::::::::::::Co:::::::::::::::oB:::::::::::::::::B R::::::R     R:::::R  A:::::A               A:::::A  
     CCC::::::::::::C oo:::::::::::oo B::::::::::::::::B  R::::::R     R:::::R A:::::A                 A:::::A 
        CCCCCCCCCCCCC   ooooooooooo   BBBBBBBBBBBBBBBBB   RRRRRRRR     RRRRRRRAAAAAAA                   AAAAAAA

"""

SCRIPT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
RINALMO_PATH = os.path.join(PARENT_DIR, "RiNALMo")

sys.path.insert(0, str(RINALMO_PATH))
# # 현재 디렉토리에 embedder.py, predictor.py가 있다고 가정
# SCRIPT_DIR = os.path.dirname(__file__)
# sys.path.insert(0, SCRIPT_DIR)

from embedder import load_model, embed_fasta
from predictor import run_folder


import argparse
import os
from pathlib import Path

def fasta_file(path: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"File '{path}' not found.")
    if p.suffix.lower() not in ('.fa', '.fasta'):
        raise argparse.ArgumentTypeError(f"Extension of file '{path}' must be .fa or .fasta.")
    return p

# def emb_dir(path: str) -> Path:
#     p = Path(path)
#     if not p.is_dir():
#         raise argparse.ArgumentTypeError(f"'{path}' is not a directory.")
#     return p

def csv_file(path: str) -> Path:
    p = Path(path)
    # If creating a new file, you can skip checking for existence.
    # To enforce only the extension, use:
    if p.suffix.lower() != '.csv':
        raise argparse.ArgumentTypeError(f"Extension of file '{path}' must be .csv.")
    return p

def main():
    parser = argparse.ArgumentParser(
        description=ASCII_BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "fasta",
        type=fasta_file,
        help="Input FASTA file path (only .fa/.fasta allowed)"
    )
    parser.add_argument(
        "emb_dir",
        # type=emb_dir,
        help="Directory where embedding (.pt) files will be stored"
    )
    parser.add_argument(
        "csv_path",
        type=csv_file,
        help="CSV file to save results (.csv only)"
    )
    parser.add_argument(
        "device",
        nargs="?",
        choices=["cuda", "cpu"],
        default=None,
        help="Compute device (cuda or cpu). Auto-detected if not specified"
    )

    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1) Embedding 생성
    model, alphabet = load_model()
    embed_fasta(args.fasta, args.emb_dir, model, alphabet)

    # 2) 예측 수행
    run_folder(args.emb_dir, args.csv_path, device)


if __name__ == "__main__":
    main()
