import time

with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
    script = f.readlines()
    model_name = script[58].split("'")[1].split('/')[1]
    batch_size = int(script[67].split('= ')[1])
    nb_epochs = int(script[66].split('= ')[1])

now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(" ", "_")
    
log_filename = f'logs/{now}_{model_name}_{batch_size}_bs_{nb_epochs}_epochs.log'

with open('output/ALTEGRAD.log', 'r') as f:
    old_file = f.readlines()

new_file = []
for line in old_file:
    if len(line) > 1 and not 'warning' in line:
        new_file.append(line.split('data":')[1][1:-5])

with open(log_filename, 'w') as f:
    for line in new_file:
        f.write(line+'\n')