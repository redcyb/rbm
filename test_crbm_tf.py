import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from models.rbm import CRBM, FRBM
from models.tfrbm import FuzzyBerBerRBM
from models.tfrbm.crbm import CrispBerBerRBM
from util import test_reconstruction

mnist = input_data.read_data_sets('mnist/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels


MODEL = CrispBerBerRBM
VISIBLE = 784
HIDDEN = 64
EPOCHS = 10

rbm = CrispBerBerRBM(VISIBLE, HIDDEN, use_tqdm=True)

rbm.fit(x_train, n_epoches=EPOCHS, verbose=True)

result = rbm.reconstruct(x_test)

print(result)
