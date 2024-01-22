rm -R Kaggle_dataset/data/processed
python3 -W ignore preprocess.py $1 $2 $3 $4

# change dataset files:
kaggle datasets version -p Kaggle_dataset -m "Update" --dir-mode zip