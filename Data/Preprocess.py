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
