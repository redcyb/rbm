from datetime import datetime

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from models.rbm import FRBM
from models.rbm import CRBM

mnist = input_data.read_data_sets('mnist/', one_hot=True)
mnist_images = mnist.train.images


def get_instance(model, n_hidden=64):
    settings = {"n_visible": 784, "n_hidden": n_hidden,
                "learning_rate": 0.01, "momentum": 0.95,
                "use_tqdm": True}
    if model == "rbm":
        return CRBM(**settings)
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
VISIBLE = 784
HIDDEN = 128
EPOCHS = 51

bbrbm = get_instance(MODEL, n_hidden=HIDDEN)

# bbrbm.save_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{0}.json")
# bbrbm.load_weights(f"./weights/{MODEL}___hid_{HIDDEN}___ep_{0}.json")

# bbrbm.save_weights(f"./weights/{MODEL}___{VISIBLE}x{HIDDEN}___ep_{0}.json")
try:
    bbrbm.load_weights(f"./weights/{MODEL}___{VISIBLE}x{HIDDEN}___ep_{0}.json")
except:
    print("No old weights")

start = datetime.now()
errors = train_rbm(bbrbm, EPOCHS)
finish = datetime.now()
spent = (finish - start).seconds

print(f"Time spent: {spent} sec")

bbrbm.save_weights(f"./weights/{MODEL}___{VISIBLE}x{HIDDEN}___ep_{EPOCHS}_2.json")
bbrbm.save_details(f"./details/{MODEL}___{VISIBLE}x{HIDDEN}___ep_{EPOCHS}_2.json", {"training_time (s)": spent, "errors": errors})
