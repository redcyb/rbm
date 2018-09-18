from datetime import datetime

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from models.frbm import FRBM
from models.rbm import RBM

mnist = input_data.read_data_sets('mnist/', one_hot=True)
mnist_images = mnist.train.images


def get_instance(model, n_hidden=64, seed=12345):
    settings = {"n_visible": 784, "n_hidden": n_hidden,
                "learning_rate": 0.01, "momentum": 0.95,
                "use_tqdm": True, "seed": seed}
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


MODEL = "frbm"
# MODEL = "rbm"
HIDDEN = 64
EPOCHS = 50
SEED = 93459

bbrbm = get_instance(MODEL, n_hidden=HIDDEN, seed=SEED)

# bbrbm.save_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{0}")
bbrbm.load_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{0}")
#
start = datetime.now()
errors = train_rbm(bbrbm, EPOCHS)
finish = datetime.now()
spent = (finish - start).seconds

print(f"Time spent: {spent} sec")

bbrbm.save_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{EPOCHS}")
bbrbm.save_details(f"./details/{MODEL}___hid_{HIDDEN}___ep_{EPOCHS}", {"training_time (s)": spent, "errors": errors})
