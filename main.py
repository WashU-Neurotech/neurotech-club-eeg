# found https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
# apparently only matters for macos, so uncomment if on macos
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
from network import *

np.random.seed(1)
tf.random.set_seed(1)

with open("config.json") as f:
    config = json.load(f)

train_network(config)

