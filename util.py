import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network._base import relu


def xavier_init(in_size, out_size, const=1.0):
    k = const * np.sqrt(6.0 / (in_size + out_size))
    return np.random.uniform(-k, k, (in_size, out_size))


def sample_bernoulli(probs):
    return relu(np.sign(probs - np.random.uniform(size=probs.shape)))


def sample_bernoulli_batch(probs):
    return relu(np.sign(probs - np.random.uniform(size=probs.shape)))


def sample_gaussian(x, sigma):
    return x + np.random.normal(0.0, sigma, x.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def show_digit_image(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()


def save_original_digit_image(x, name, label):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.savefig(f"./images/{name}___{label}___z_orig.png")


def save_digit_image(x, name, label, config_str):
    full_path = f"./images/{name}___{label}___{config_str}___recon.png"
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.savefig(full_path)


def show_original_digit_image(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()


def test_reconstruction(mnist_img_num, xs, ys, instance, config_str):
    image = xs[mnist_img_num]
    image_rec = instance.reconstruct(image)

    save_original_digit_image(image, mnist_img_num, label_to_str(ys[mnist_img_num]))

    save_digit_image(image_rec, mnist_img_num, label_to_str(ys[mnist_img_num]), config_str)

    # show_original_digit_image(image)
    # show_digit_image(image_rec)


def label_to_str(label):
    return str(list(label).index(1))
