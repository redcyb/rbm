import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from frbm import FRBM
from rbm import RBM
from util import save_digit_image, save_original_digit_image, show_digit_image, show_original_digit_image

mnist = input_data.read_data_sets('mnist/', one_hot=True)
mnist_images = mnist.train.images


def get_instance(model, n_hidden=64):
    settings = {"n_visible": 784, "n_hidden": n_hidden, "learning_rate": 0.01, "momentum": 0.95, "use_tqdm": True}
    if model == "rbm":
        return RBM(**settings)
    if model == "frbm":
        return FRBM(**settings)
    return None


def train_rbm(instance, epochs, show=False):
    all_errs, epochs_mean_errors = instance.fit(mnist_images, n_epoches=epochs, batch_size=10)

    if show:
        try:
            lim = len(all_errs) // 3
            plt.plot(all_errs[:lim])
            plt.show()
            plt.clf()
            plt.plot(all_errs[lim:])
            plt.show()
        except:
            pass
    return epochs_mean_errors


def test_reconstruction(instance, mnist_img_num, **kwargs):
    image = mnist_images[mnist_img_num]
    image_rec = instance.reconstruct(image)
    save_original_digit_image(image, mnist_img_num)
    save_digit_image(image_rec, mnist_img_num, **kwargs)
    # show_original_digit_image(image)
    # show_digit_image(image_rec, mnist_img_num, **kwargs)


def test_restoring_partial(instance, mnist_img_num, **kwargs):
    image = mnist_images[mnist_img_num]
    mid = 550
    image1 = np.hstack((image[:mid], np.array([0 for i in range(len(image) - mid)])))
    image_rec = instance.reconstruct(image1)
    # save_original_digit_image(image, mnist_img_num)
    show_original_digit_image(image)
    show_digit_image(image_rec, mnist_img_num, **kwargs)


def test_load_and_reconstruct(instance, filename, num=None):
    instance.load_weights(filename)
    if num is not None:
        test_reconstruction(instance, num)


MODEL = "frbm"
# MODEL = "rbm"
HIDDEN = 64
EPOCHS = 10

bbrbm = get_instance(MODEL, n_hidden=HIDDEN)
bbrbm.load_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{EPOCHS}")

# test_reconstruction(bbrbm, 0, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 2, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 3, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 4, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 5, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 6, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 7, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 8, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 9, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 10, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 11, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 12, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 13, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 14, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 15, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 16, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 17, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 18, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 19, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)

# test_reconstruction(bbrbm, 1000, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1001, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1002, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1003, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1004, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1005, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1006, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
test_reconstruction(bbrbm, 1007, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1008, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
# test_reconstruction(bbrbm, 1009, model=MODEL, hidden=HIDDEN, epochs=EPOCHS)
