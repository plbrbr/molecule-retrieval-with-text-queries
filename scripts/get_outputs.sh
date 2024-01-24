kaggle kernels output manalakhannouss/ALTEGRAD -p output
python3 utils/clean_logs.py
rm -R output

python3 -W ignore utils/correct_train.py