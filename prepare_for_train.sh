rm -R Kaggle_dataset/data/processed
python3 -W ignore utils/preprocess.py $1 $2 $3
kaggle datasets version -p Kaggle_dataset -m "Update" --dir-mode zip