#!/usr/bin/env python3
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

from CoBRA.embedder import load_model, embed_fasta
from CoBRA.predictor import run_folder
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

def csv_file(path: str) -> Path:
    p = Path(path)

    if p.suffix.lower() != '.csv':
        raise argparse.ArgumentTypeError(f"Extension of file '{path}' must be .csv.")
    return p

def main():
    print(f"{ASCII_BANNER} is running!")
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

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, alphabet = load_model()
    embed_fasta(args.fasta, args.emb_dir, model, alphabet)

    run_folder(args.emb_dir, args.csv_path, device)


if __name__ == "__main__":
    main()
