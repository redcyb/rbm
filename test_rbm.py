from datetime import datetime

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from frbm import FRBM
from rbm import RBM
from util import save_digit_image

mnist = input_data.read_data_sets('mnist/', one_hot=True)
mnist_images = mnist.train.images


def get_instance(model, n_hidden=64):
    settings = {"n_visible": 784, "n_hidden": n_hidden, "learning_rate": 0.01, "momentum": 0.95, "use_tqdm": True}
    if model == "rbm":
        return RBM(**settings)
    if model == "frbm":
        return FRBM(**settings)
    return None


def train_rbm(instance, epochs):
    errs = instance.fit(mnist_images, n_epoches=epochs, batch_size=10)

    try:
        lim = len(errs) // 3
        plt.plot(errs[:lim])
        plt.show()
        plt.clf()
        plt.plot(errs[lim:])
        plt.show()
    except:
        pass


def test_reconstruction(instance, mnist_img_num):
    image = mnist_images[mnist_img_num]
    image_rec = instance.reconstruct(image)
    save_digit_image(image, mnist_img_num, MODEL, True)
    save_digit_image(image_rec, mnist_img_num, MODEL, False)


def test_load_and_reconstruct(instance, filename, num=None):
    instance.load_weights(filename)
    if num is not None:
        test_reconstruction(instance, num)


MODEL = "frbm"
# MODEL = "rbm"
HIDDEN = 128
EPOCHS = 20

bbrbm = get_instance(MODEL, n_hidden=HIDDEN)
bbrbm.save_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{0}")

start = datetime.now()
train_rbm(bbrbm, EPOCHS)
finish = datetime.now()
spent = (finish - start).seconds

print(f"Time spent: {spent} sec")

bbrbm.save_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{EPOCHS}")
bbrbm.save_details(f"./details/{MODEL}___hid_{HIDDEN}___ep_{EPOCHS}",
                   {"training_time (s)": spent})

# test_load_and_reconstruct(bbrbm, f"./weights/{MODEL}___hid_{HIDDEN}___ep_{EPOCHS}")
#
# test_reconstruction(bbrbm, 0)
# test_reconstruction(bbrbm, 1)
# test_reconstruction(bbrbm, 2)
# test_reconstruction(bbrbm, 3)
# test_reconstruction(bbrbm, 4)
# test_reconstruction(bbrbm, 5)
# test_reconstruction(bbrbm, 6)
# test_reconstruction(bbrbm, 7)
# test_reconstruction(bbrbm, 8)
# test_reconstruction(bbrbm, 9)
