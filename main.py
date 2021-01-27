import json
from data import *
from network import *

np.random.seed(1)
tf.random.set_seed(1)

with open("config.json") as f:
    config = json.load(f)

train_data, val_data = get_data(config)

print()
print(train_data[0].shape)
print(val_data[0].shape)

train_network(train_data, val_data)

