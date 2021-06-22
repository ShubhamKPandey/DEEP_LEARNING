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

print(np.array(tokenizer.texts_to_sequences([shakespeare_text])))
print(np.array(tokenizer.sequences_to_texts([[34, 5, 6, 23, 7]])))

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
print(encoded)