import tensorflow as tf
from tensorflow import keras
import random
import numpy as np



class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_  = np.mean(data_sample, axis = 0, keepdims = True)
        self.stds_ = np.std(data_sample, axis = 0, keepdims = True)

    def call(self, inputs):
        return (inputs - self.means_)/ (self.stds_ + keras.backend.epsilon())

        
vocab = [ "<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND" ]
indices = tf.range(len(vocab), dtype = tf.int64) 
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)


categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
print(cat_indices)
cat_one_hot = tf.one_hot(cat_indices, depth = len(vocab) + num_oov_buckets)
print(cat_one_hot)

embedding_dim  = 2
embed_init  = tf.random.uniform([len(vocab) + num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)
print(embedding_matrix)

categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
print(cat_indices)
print(tf.nn.embedding_lookup(embedding_matrix , cat_indices))

regular_inputs = keras.layers.Input(shape = [8])
categories = keras.layers.Input(shape = [], dtype = tf.string)
cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(categories)
cat_embed = keras.layers.Embedding(input_dim = 6, output_dim = 2)(cat_indices)
encoded_inputs = keras.layers.concatenate([regular_inputs, cat_embed])
outputs = keras.layers.Dense(1)(encoded_inputs)
model = keras.models.Model(inputs = [regular_inputs, categories], outputs = [outputs])

