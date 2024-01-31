kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
tensorboard --logdir=runs --port=3333