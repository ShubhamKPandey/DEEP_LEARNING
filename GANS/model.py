import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] /255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

stacked_encoder = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(100, activation="selu"),
keras.layers.Dense(30, activation="selu"),
])
stacked_decoder = keras.models.Sequential([
keras.layers.Dense(100, activation="selu", input_shape=[30]),
keras.layers.Dense(28 * 28, activation="sigmoid"),
keras.layers.Reshape([28, 28])
])
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5))

history = stacked_ae.fit(X_train, X_train, epochs=20,validation_data=(X_valid,X_valid))
i = 0
def plot_imagea(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.savefig('./image{}'.format(0))
def plot_imageb(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.savefig('./image{}'.format(1))

def show_reconstructions(model, n_images=5):
    reconstructions = model.predict(X_valid[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_imagea(X_valid[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_imageb(reconstructions[image_index])
show_reconstructions(stacked_ae)