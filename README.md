# CoBRA
It only works in a GPU environment.
## Clone the repo
```bash
git clone https://github.com/WonKyeongJang/CoBRA
cd ./CoBRA
git clone https://github.com/lbcb-sci/RiNALMo
```

### Create conda environment for CoBRA
```bash
conda env create -f ./environment.yml
conda activate CoBRA
```
### Usuage
```bash
python main.py sample/sample.fasta sample/embeddings sample/sample.csv
```
sample/sample.fasta : "Input FASTA file path (only .fa/.fasta allowed)"\
sample/embeddings : "Directory where embedding (.pt) files will be stored"\
sample/sample.csv : "CSV file to save results (.csv only)"
