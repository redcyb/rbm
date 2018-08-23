from datetime import datetime

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from frbm import FRBM
from rbm import RBM

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


# MODEL = "frbm"
MODEL = "rbm"
HIDDEN = 64
EPOCHS = 10

bbrbm = get_instance(MODEL, n_hidden=HIDDEN)

bbrbm.save_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{0}")
bbrbm.load_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{0}")

frbm = get_instance("frbm", n_hidden=HIDDEN)

frbm.w_l = bbrbm.w - bbrbm.shift_w
frbm.w_r = bbrbm.w + bbrbm.shift_w

frbm.bv_l = bbrbm.bv - bbrbm.shift_bv
frbm.bv_r = bbrbm.bv + bbrbm.shift_bv

frbm.bh_l = bbrbm.bh - bbrbm.shift_bh
frbm.bh_r = bbrbm.bh + bbrbm.shift_bh

frbm.save_weights(f"./weights/frbm___hid_{HIDDEN}___ep_{0}")

print()

# print(f"Time spent: {spent} sec")

# start = datetime.now()
# errors = train_rbm(bbrbm, EPOCHS)
# finish = datetime.now()
# spent = (finish - start).seconds
#
# print(f"Time spent: {spent} sec")
#
# bbrbm.save_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{EPOCHS}")
# bbrbm.save_details(f"./details/{MODEL}___hid_{HIDDEN}___ep_{EPOCHS}", {"training_time (s)": spent, "errors": errors})
