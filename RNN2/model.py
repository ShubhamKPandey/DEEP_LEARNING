import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)