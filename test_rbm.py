import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from models.rbm import CRBM, FRBM
from util import test_reconstruction

mnist = input_data.read_data_sets('mnist/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels


def train_rbm(instance, epochs, show=False):
    all_errs, epochs_mean_errors = instance.fit(x_train, n_epoches=epochs, batch_size=10)

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


MODEL = CRBM
VISIBLE = 784
HIDDEN = 392
EPOCHS = 50

bbrbm = MODEL(n_visible=VISIBLE, n_hidden=HIDDEN, learning_rate=0.01, momentum=0.95, use_tqdm=True)
# bbrbm.load_weights(f"./weights/{MODEL.__name__.lower()}___{VISIBLE}x{HIDDEN}___ep_{EPOCHS}.json")

train_rbm(bbrbm, EPOCHS, show=True)

for i in [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010
]:
    test_reconstruction(i, x_test, y_test, bbrbm,
                        f"{MODEL.__name__.lower()}___{VISIBLE}x{HIDDEN}___ep_{EPOCHS}")
