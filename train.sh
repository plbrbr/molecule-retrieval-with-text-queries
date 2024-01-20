# preprocess data
rm -R Kaggle_dataset/data/preprocessed
python3 preprocess.py $1

# change dataset files:
kaggle datasets version -p Kaggle_dataset -m "Update" --dir-mode zip

# push and run script:
kaggle kernels push -p Kaggle_kernel

# get outputs of a kernel:
kaggle kernels output alexlutt/ALTEGRAD -p output

# submit:
kaggle competitions submit altegrad-2023-data-challenge -f output/submission.csv