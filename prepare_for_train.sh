# preprocess data
rm -R Kaggle_dataset/data/processed
rm -R output
python3 -W ignore preprocess.py $1

# change dataset files:
kaggle datasets version -p Kaggle_dataset -m "Update" --dir-mode zip