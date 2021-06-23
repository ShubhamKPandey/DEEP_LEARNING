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




