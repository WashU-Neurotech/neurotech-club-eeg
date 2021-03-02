from data import *
from model import *
from datetime import datetime
import json

def train_network(config):
    model_type = config["model"]["model_type"]
    epochs = config["train"]["epochs"]
    learning_rate = config["train"]["learning_rate"]
    kernel_regularizer_type = config["model"][model_type]["kernel_regularize"]["type"]
    kernel_regularizer_amount = config["model"][model_type]["kernel_regularize"]["amount"]

    train_data, val_data = get_data(config)

    print()
    print(train_data[0].shape)
    print(val_data[0].shape)

    if model_type == "baby_binary":
        model = baby_binary(train_data[0].shape[1], 
                        num_hidden_layer_nodes=config["model"]["baby_binary"]["num_hidden_layer_nodes"],
                        kernel_regularizer_type=kernel_regularizer_type,
                        kernel_regularizer_amount=kernel_regularizer_amount)
    elif model_type == "dnn_binary":
        model = dnn_binary(train_data[0].shape[1],
                        hidden_layers=config["model"]["dnn_binary"]["hidden_layers"],
                        kernel_regularizer_type=kernel_regularizer_type,
                        kernel_regularizer_amount=kernel_regularizer_amount)
    else:
        quit("Check model name in config. {} does not exist as a model.".format(model_type))

    print(model.summary())

    model_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Set Tensorboard callback
    tensorboard_logdir = config["settings"]["root_path"] + "results\\tensorboard_callbacks\\{}\\".format(model_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir)

    print("\n--------------- Saving config_train ---------------\n")
    with open(config["settings"]["root_path"] + "results\\saved_models\\configs\\{}.json".format(model_name), "w") as f:
        json.dump(config, f)

    print("\n--------------- Saving Model Architecture ---------------\n")
    model_config = model.to_json()
    with open(config["settings"]["root_path"] + "results\\saved_models\\architectures\\{}.json".format(model_name), "w") as f:
        f.write(model_config)


    print("\n\n--------------- Training Starting ---------------\n\n")

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
    
    history = model.fit(x=train_data[0], y=train_data[1],
                    validation_data=val_data,
                    batch_size=32,
                    verbose=1, # set to 0 for no epoch updates; 1 for updates
                    epochs=epochs,
                    callbacks=[tensorboard_callback]
                    )

    print("\n\n--------------- Training Complete ---------------\n\n")

    model.save_weights(config["settings"]["root_path"] + "results\\saved_models\\end_weights\\{}\\end_weights".format(model_name,model_name))