with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
    script = f.readlines()
    model_name = script[58].split("'")[1].split('/')[1]
    batch_size = int(script[66].split('= ')[1])
    nb_epochs = int(script[67].split('= ')[1])
    
log_filename = f'logs/{model_name}_bs_{batch_size}_epochs_{nb_epochs}.log'

with open('output/ALTEGRAD.log', 'r') as f:
    old_file = f.readlines()

new_file = []
for line in old_file:
    if len(line) > 1 and not 'arning' in line:
        new_file.append(line.split('data":')[1][1:-5])

with open(log_filename, 'w') as f:
    for line in new_file:
        f.write(line+'\n')