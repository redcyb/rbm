import matplotlib.pyplot as plt

from models.rbm import CRBM, FRBM
from util import test_reconstruction, show_filters
from vizualize_hidden import tile_raster_images_grayscale

import numpy

from sklearn.model_selection import train_test_split


def build_nefclass(inputs, n_outputs):
    print(inputs)
    print(n_outputs)


filename = "./data/iris/iris_prepared_one_hot.csv"
dataset = numpy.loadtxt(filename, delimiter=",")

xs = dataset[:, :-3]
xs = (xs - xs.min(0)) / xs.ptp(0)
ys = dataset[:, -3:]

x_train, x_test, y_train, y_test = train_test_split(xs, ys, shuffle=True)


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
DATASET = "iris"
VISIBLE = 4
HIDDEN = 32
EPOCHS = 51

bbrbm = MODEL(n_visible=VISIBLE, n_hidden=HIDDEN, learning_rate=0.01, momentum=0.95, use_tqdm=True)

try:
    bbrbm.load_weights(f"./weights/{DATASET}___rbm___{VISIBLE}x{HIDDEN}___ep_{EPOCHS}_2.json")
except:
    pass
    train_rbm(bbrbm, EPOCHS, show=True)
# train_rbm(bbrbm, EPOCHS, show=True)

# for i in [
#     # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#     # 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#     # 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010
# ]:
#     test_reconstruction(
#         i, x_test, y_test, bbrbm,
#         f"rbm___{VISIBLE}x{HIDDEN}___ep_{EPOCHS}_2"
#     )

# filters = tile_raster_images_grayscale(bbrbm.w.T, (28, 28), (8, 8))

# show_filters(filters)
