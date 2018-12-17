import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from models.rbm import CRBM, FRBM
from models.tfrbm import FuzzyBerBerRBM
from util import test_reconstruction

mnist = input_data.read_data_sets('mnist/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels


MODEL = FuzzyBerBerRBM
VISIBLE = 784
HIDDEN = 392
EPOCHS = 101

rbm = FuzzyBerBerRBM(VISIBLE, HIDDEN, use_tqdm=True)

rbm.fit(x_train, n_epoches=EPOCHS, verbose=True)

result = rbm.reconstruct(x_test)

print(result)
