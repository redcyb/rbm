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


def save_digit_image(x, name, model, orig):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.savefig(f"./images/{model}/{name}___{'orig' if orig else 'recon'}___{model}.png")
