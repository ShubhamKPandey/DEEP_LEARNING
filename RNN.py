import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_time_series(batch_size = 1 , n_time_steps = 50, n = 1):
    freq1, freq2, offset1 , offset2 = random.rand(4, batch_size, 1)
    timesteps = np.linspace(0.0, 1.0, num = n_time_steps)

    series = 0.5*np.sin(
        (timesteps-offset1)*(freq1*10 + 10)
    )
    series += 0.2*np.sin(
        (timesteps-offset2)*(freq2*20 + 20)
    )
    series += 0.1*random.rand(
        batch_size,n_time_steps
    )
    return series.astype(np.float32)


steps = 50
batch_size = 10000
dataset = np.array(generate_time_series(batch_size,n_time_steps = steps + 1))

X_train, y_train = dataset[0:7000,0:steps], dataset[0:7000, -1]
X_valid, y_valid = dataset[7000:9000, 0:steps], dataset[7000:9000, -1]
X_test, y_test = dataset[9000:, 0:steps], dataset[9000:, -1]
print(y_valid.shape, y_train.shape)

input = tf.keras.layers.Input(shape = (steps,1))
x = tf.keras.layers.Flatten()(input)
y = tf.keras.layers.Dense(1)(x)


model = tf.keras.Model(inputs = [input], outputs = [y])
model.compile(loss = "mean_squared_error", optimizer = "Adam")
history = model.fit(X_train, y_train,epochs = 20,validation_data=(X_valid, y_valid))




