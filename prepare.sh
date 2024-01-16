conda create -n ml python=3.9
conda activate ml
pip install -r requirements.txt

export KAGGLE_USERNAME=
export KAGGLE_KEY=

mkdir data
kaggle competitions download -c altegrad-2023-data-challenge -p ./data

git config --global user.email ""
git config --global user.name ""