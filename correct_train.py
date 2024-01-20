import sys

model_name = sys.argv[1]

with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
    new_file = f.readlines()
    try:
        line_id = new_file.index(f"    model_name = '{model_name}'\n")
        new_file[line_id] = "    model_name = 'bert-base-uncased'\n"
    except:
        print("Model name could not be fixed")
        exit()

with open('Kaggle_kernel/kaggle-script.py', 'w') as f:
    f.writelines(new_file)
    print("Model name successfully fixed")
