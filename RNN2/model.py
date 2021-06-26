import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

shakespeare_url = "https://homl.info/shakespeare" # shortcut URL
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level = True)
tokenizer.fit_on_texts([shakespeare_text])

print(tokenizer.document_count)

dataset_size = tokenizer.document_count
max_id = len(tokenizer.word_index)

print(np.array(tokenizer.sequences_to_texts([[34, 5, 6, 23, 7]])))

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
print(encoded)

train_size = dataset_size*90//100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_length  = n_steps + 1
dataset = dataset.window(window_length, shift = 1, drop_remainder = True )

dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth = max_id), Y_batch))

dataset = dataset.prefetch(1)

model =  keras.models.Sequential([
    keras.layers.GRU(128, return_sequences =True, input_shape = [None, max_id],
    dropout  =0.2, recurrent_dropout = 0.2
    ),
    keras.layers.GRU(128, return_sequences = True,
    dropout = 0.2, recurrent_dropout = 0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, 
    activation = "softmax"))
]
)

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam")

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

def next_char(text, temperature = 1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0,-1:,: ]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples =1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, n_chars = 50, temperature = 1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


#Stateful
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
dataset = dataset.window(window_length, shift = n_steps, drop_remainder = True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
dataset = dataset.batch(1)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:,1:]))
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth = max_id), Y_batch)
)
dataset = dataset.prefetch(1)

tf.train.Dataset.zip(dataset).map(lambda *windows:
tf.stack(windows))


model = keras.models.Sequential([
keras.layers.GRU(128, return_sequences=True, stateful=True,
dropout=0.2, recurrent_dropout=0.2,
batch_input_shape=[batch_size, None, max_id]),
keras.layers.GRU(128, return_sequences=True, stateful=True,
dropout=0.2, recurrent_dropout=0.2),
keras.layers.TimeDistributed(keras.layers.Dense(max_id,
activation="softmax"))
])

class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(dataset, epochs=50, callbacks=[ResetStatesCallback()])