from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from models.dbn import CrispDBN, FuzzyDBN
from models.mlp import Perceptron

mnist = input_data.read_data_sets('mnist/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

# (x_train, y_train), (x_test, y_test) = mnist_ds.load_data()

MODEL = CrispDBN
EPOCHS = 100
SEED = np.random.seed(123)
LAYERS = (784, 64, 10)

# dbn = MODEL(layers=LAYERS[:-1])
# dbn.rbms[0].load_weights(f"./weights/{MODEL.RBM_MODEL.__name__.lower()}___784x392___ep_50.json")
# dbn.rbms[1].load_weights(f"./weights/{MODEL.RBM_MODEL.__name__.lower()}___392x64___ep_50.json")

# dbn.rbms[0].load_weights(f"./weights/rbm___hid_64___ep_50")
# dbn.train_rbms(x_train, epochs=EPOCHS, train_from=1)
# dbn.save_weights(f"./weights/{MODEL}___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}")

perceptron = Perceptron(layers=LAYERS)

# perceptron.w[0] = dbn.rbms[0].w.T
# perceptron.b[0] = dbn.rbms[0].bh
#
# perceptron.w[1] = dbn.rbms[1].w.T
# perceptron.b[1] = dbn.rbms[1].bh

# perceptron.load_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___ep_300+.json")
perceptron.train_batch(x_train, y_train, report_every=1, epochs=EPOCHS, batch_size=13, learning_rate=0.01,
#                        only_last=True
                       )
perceptron.save_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___ep_300+.json")
# perceptron.load_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}.json")

perceptron.predict(x_test, y_test, save_result=False)

pass
