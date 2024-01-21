with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
    new_file = f.readlines()
    line_id = 52
    new_file[line_id] = "    model_name = 'bert-base-uncased'\n"

with open('Kaggle_kernel/kaggle-script.py', 'w') as f:
    f.writelines(new_file)
    print("Model name successfully fixed")
