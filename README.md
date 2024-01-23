# Kaggle ALTeGRaD 2023-2024
Data Challenge - Molecule Retrieval with Natural Language Queries

## Setup requirements

Please follow the following instructions to install the required packages:

```bash
python -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## New training session

```bash
sh prepare_for_run.sh model_name nb_epochs batch_size
# wait until dataset is loaded on Kaggle
sh train.sh
# wait until run finished
sh get_outputs.sh
sh submit.sh
```