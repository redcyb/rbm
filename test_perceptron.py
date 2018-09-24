from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from models.dbn import DBN
from models.mlp import Perceptron

mnist = input_data.read_data_sets('mnist/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

# (x_train, y_train), (x_test, y_test) = mnist_ds.load_data()

MODEL = "DBN"
EPOCHS = 0
SEED = np.random.seed(1)
LAYERS = (784, 300, 10)

# dbn = DBN(layers=LAYERS)
# dbn.rbms[0].load_weights(f"./weights/rbm___hid_64___ep_50")
# dbn.train_rbms(x_train, epochs=EPOCHS, train_from=1)
# dbn.save_weights(f"./weights/{MODEL}___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}")

perceptron = Perceptron(layers=LAYERS)
# perceptron.load_weights(f"./weights/{MODEL}___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}")
# perceptron.load_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}")
# perceptron.load_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___ep_60_last")
perceptron.train_batch(x_train, y_train, report_every=1, epochs=100, batch_size=100, learning_rate=0.01)
perceptron.save_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___ep_160_last")
perceptron.predict(x_test[:10], y_test[:10])

pass
