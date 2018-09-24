import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from models.frbm import FRBM
from models.rbm import RBM
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


# def test_restoring_partial(instance, mnist_img_num, **kwargs):
#     image = x_test[mnist_img_num]
#     mid = 550
#     image1 = np.hstack((image[:mid], np.array([0 for i in range(len(image) - mid)])))
#     image_rec = instance.reconstruct(image1)
#     # save_original_digit_image(image, mnist_img_num)
#     show_original_digit_image(image)
#     show_digit_image(image_rec, mnist_img_num, **kwargs)

MODEL = FRBM
VISIBLE = 784
HIDDEN = 392
EPOCHS = 50

bbrbm = MODEL(n_visible=VISIBLE, n_hidden=HIDDEN, learning_rate=0.01, momentum=0.95, use_tqdm=True)
bbrbm.load_weights(f"./weights/{MODEL.__name__.lower()}___{VISIBLE}x{HIDDEN}___ep_{EPOCHS}.json")

for i in [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010
]:
    test_reconstruction(i, x_test, y_test, bbrbm,
                        f"{MODEL.__name__.lower()}___{VISIBLE}x{HIDDEN}___ep_{EPOCHS}")
