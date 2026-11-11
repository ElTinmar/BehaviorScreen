python paper_reference.py download \
    --output-dir paper_data

python paper_reference.py extract \
    --data-dir paper_data \
    --output paper_data/paper_reference.csv

python paper_reference.py build \
    --reference-csv paper_data/paper_reference.csv \
    --output paper_data/paper_reference.joblib

python paper_reference.py plot \
    --reference-csv paper_data/paper_reference.csv \
    --model paper_data/paper_reference.joblib \
    --output paper_data/reference_umap.png