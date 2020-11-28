import json
from data import *

with open("config.json") as f:
    config = json.load(f)

train_data, val_data = get_data()

print(train_data[0].shape)
print(val_data[0].shape)