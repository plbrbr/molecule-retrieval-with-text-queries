with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
    new_file = f.readlines()
    new_file[55] = "    model_name = 'bert-base-uncased'\n"
    new_file[63] = "    nb_epochs = 10\n"
    new_file[64] = "    batch_size = 32\n"
    new_file[65] = "    learning_rate = 2e-5'\n"

with open('Kaggle_kernel/kaggle-script.py', 'w') as f:
    f.writelines(new_file)
    print("Script successfully fixed")
