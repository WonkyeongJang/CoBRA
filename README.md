# Clone the repo.
git clone https://github.com/WonKyeongJang/CoBRA
cd ./CoBRA
git clone https://github.com/lbcb-sci/RiNALMo

# create conda environment for CoBRA
conda env create -f ./environment.yml
conda activate CoBRA

# Usuage
python ./main.py ./sample.fasta  ./embeddings ./sample.csv
