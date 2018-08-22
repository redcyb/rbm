import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network._base import relu


def xavier_init(in_size, out_size, const=1.0):
    k = const * np.sqrt(6.0 / (in_size + out_size))
    return np.random.uniform(-k, k, (in_size, out_size))


def sample_bernoulli(probs):
    return relu(np.sign(probs - np.random.uniform(size=probs.shape)))


def sample_gaussian(x, sigma):
    return x + np.random.normal(0.0, sigma, x.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save_digit_image(x, name, **kwargs):
    model = kwargs.get("model")
    epochs = kwargs.get("epochs")
    hidden = kwargs.get("hidden")

    full_path = f"./images/{name}"

    if model:
        full_path = f"{full_path}___{model}"
    if hidden:
        full_path = f"{full_path}___hid_{hidden}"
    if epochs:
        full_path = f"{full_path}___epo_{epochs}"

    full_path = f"{full_path}___recon.png"

    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.savefig(full_path)


def save_original_digit_image(x, name):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.savefig(f"./images/{name}___orig.png")
