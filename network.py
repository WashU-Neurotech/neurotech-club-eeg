from model import *

def train_network(train_data, val_data):

    model = baby_model(train_data[0].shape[1])
    print(model.summary())
    print("HI")

    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
    
    history = model.fit(x=train_data[0], y=train_data[1],
                    validation_data=val_data,
                    batch_size=32,
                    verbose=1, # set to 0 for no epoch updates; 1 for updates
                    epochs=100)
