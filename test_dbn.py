import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from models.dbn import CrispDBN, FuzzyDBN
from util import test_reconstruction

mnist = input_data.read_data_sets('mnist/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

# MODEL = CrispDBN
MODEL = FuzzyDBN
EPOCHS = 101
SEED = np.random.seed(123)
LAYERS = (784, 392, 64, 10)

dbn = MODEL(layers=LAYERS)

# dbn.rbms[0].load_weights(f"./weights/{MODEL.RBM_MODEL.__name__.lower()}___784x392___ep_101.json")
# dbn.rbms[1].load_weights(f"./weights/{MODEL.RBM_MODEL.__name__.lower()}___392x64___ep_101.json")
# dbn.rbms[2].load_weights(f"./weights/{MODEL.RBM_MODEL.__name__.lower()}___64x10___ep_101.json")

# ===== TRAIN DBN

dbn.train_rbms(x_train, epochs=EPOCHS, save_result=True, train_from=0)

# ===== TEST DBN

# for i in [
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#     1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010
# ]:
#     test_reconstruction(i, x_test, y_test, dbn,
#                         f"{MODEL.__name__.lower()}___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}")
