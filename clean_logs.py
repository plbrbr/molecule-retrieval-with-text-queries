with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
    script = f.readlines()
    model_name = script[55].split("'")[1]
    batch_size = int(script[64].split('= ')[1])
    learning_rate = float(script[65].split('= ')[1])
    nb_epochs = int(script[63].split('= ')[1])

log_filename = f'logs/{model_name}_bs_{batch_size}_lr_{learning_rate}_epochs_{nb_epochs}.log'

with open('output/ALTEGRAD.log', 'r') as f:
    old_file = f.readlines()

new_file = []
for line in old_file:
    if len(line) > 1:
        new_file.append(line.split('data":')[1][1:-5])

with open(log_filename, 'w') as f:
    for line in new_file:
        f.write(line+'\n')