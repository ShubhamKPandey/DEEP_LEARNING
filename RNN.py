import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


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

class LNSimpleRNNCell(keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
        activation=None)
        self.layer_norm = keras.layers.LayerNormalization()
        self.activation = keras.activations.get(activation)
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]

steps = 50
batch_size = 10000
dataset = np.array(generate_time_series(batch_size, n_time_steps = steps + 10))

Y = np.empty([batch_size, steps, 10], dtype = float)

for i in range(1,11):
    Y[:,:,i-1] = dataset[:, i:i + steps]


X_train, y_train = dataset[0:7000,0:steps], Y[0:7000,:,:]
X_valid, y_valid = dataset[7000:9000, 0:steps], Y[7000:9000,:,:]
X_test, y_test = dataset[9000:, 0:steps], Y[9000:,:,:]
print(y_valid.shape, y_train.shape)


# input = tf.keras.layers.Input(shape = (steps,1))
# x = tf.keras.layers.SimpleRNN(20, return_sequences =True)(input)
# x = tf.keras.layers.SimpleRNN(20, return_sequences = True)(input)
# y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(x)
# model = tf.keras.Model(inputs = [input], outputs = [y])

# print(model.summary())
# model.compile(loss = "mean_squared_error", optimizer = "Adam")
# history = model.fit(X_train, y_train,epochs = 20,validation_data=(X_valid, y_valid))
# pd.DataFrame(history.history).plot(figsize = (16, 8))
# plt.grid(True)
# plt.gca().set_ylim(0,0.2)
# plt.savefig('./plot7.png')



# X_train, y_train = dataset[0:7000,0:steps], dataset[0:7000, steps:]
# X_valid, y_valid = dataset[7000:9000, 0:steps], dataset[7000:9000, steps:]
# X_test, y_test = dataset[9000:, 0:steps], dataset[9000:, steps:]
# print(y_valid.shape, y_train.shape)

# input = tf.keras.layers.Input(shape = (steps,1))
# x = tf.keras.layers.Flatten()(input)
# y = tf.keras.layers.Dense(1)(x)


input = tf.keras.layers.Input(shape = (steps,1))
x = tf.keras.layers.SimpleRNN(20, return_sequences =True)(input)
x = keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True)(x)
y = keras.layers.TimeDistributed(keras.layers.Dense(10))(x)


model = tf.keras.Model(inputs = [input], outputs = [y])
print(model.summary())
model.compile(loss = "mean_squared_error", optimizer = "Adam")
history = model.fit(X_train, y_train,epochs = 20,validation_data=(X_valid, y_valid))


pd.DataFrame(history.history).plot(figsize = (16, 8))
plt.grid(True)
plt.gca().set_ylim(0,0.2)
plt.savefig('./plot8.png')

# dataset_new = generate_time_series(batch_size, n_time_steps = steps + 10 + 1)

# X_new = dataset_new[:,0:steps]
# y_new = dataset_new[:,steps:]
# print(X_new.shape)
# print(dataset_new.shape)

# for i in range(10):
#     X = X_new[:,i:]
#     y_predict = model.predict(X).reshape(batch_size,1)
#     print(y_predict.shape)
#     print(y_predict[0,:])
#     X_new = np.concatenate([X_new, y_predict], axis = 1)
#     print(X_new.shape)

# y_predicted = X_new[:,steps:]
# print(y_predicted[0,:])
# print(y_new[0,:])


