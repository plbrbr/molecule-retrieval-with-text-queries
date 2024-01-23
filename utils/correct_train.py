with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
    new_file = f.readlines()
    new_file[60] = "    model_name = 'allenai/scibert_scivocab_cased'\n"
    new_file[68] = "    nb_epochs = 10\n"
    new_file[69] = "    batch_size = 32\n"

with open('Kaggle_kernel/kaggle-script.py', 'w') as f:
    f.writelines(new_file)
    print("Script successfully fixed")
