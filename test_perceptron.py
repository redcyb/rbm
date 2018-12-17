from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from models.dbn import CrispDBN, FuzzyDBN
from models.mlp import Perceptron
from models.rbm import CRBM

mnist = input_data.read_data_sets('mnist/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

# (x_train, y_train), (x_test, y_test) = mnist_ds.load_data()

MODEL = CrispDBN
EPOCHS = 101
SEED = np.random.seed(123)
LAYERS = (784, 392, 64, 10)

# dbn = MODEL(layers=LAYERS)

# match 43; don't match 9957
# dbn.rbms[0].load_weights(f"./weights/crbm___784x392___ep_101.json")
# dbn.rbms[1].load_weights(f"./weights/crbm___392x64___ep_101.json")
# dbn.rbms[2].load_weights(f"./weights/crbm___64x10___ep_101.json")

# Match: 825   NotMatch: 9175   from 10000
# dbn.rbms[0].load_weights(f"./weights/frbm___def___784x392___ep_101.json")
# dbn.rbms[1].load_weights(f"./weights/frbm___def___392x64___ep_101.json")
# dbn.rbms[2].load_weights(f"./weights/frbm___def___64x10___ep_101.json")

# dbn.rbms[0].load_weights(f"./weights/rbm___hid_64___ep_50")
# dbn.train_rbms(x_train, epochs=EPOCHS, train_from=1)
# dbn.save_weights(f"./weights/{MODEL}___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}")

perceptron = Perceptron(layers=LAYERS)

# perceptron.w[0] = dbn.rbms[0].w.T
# perceptron.b[0] = dbn.rbms[0].bh
#
# perceptron.w[1] = dbn.rbms[1].w.T
# perceptron.b[1] = dbn.rbms[1].bh
#
# perceptron.w[2] = dbn.rbms[2].w.T
# perceptron.b[2] = dbn.rbms[2].bh

perceptron.load_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___dbn_101___frbm_to_crbm.json")
perceptron.train_batch(x_train, y_train, report_every=1, epochs=EPOCHS, batch_size=10, learning_rate=0.01,
                       # only_last=True
                       )

perceptron.save_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___dbn_101___frbm_to_crbm___tune.json")

# perceptron.save_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}.json")
# perceptron.load_weights(f"./weights/MLP___{'-'.join([str(l) for l in LAYERS])}___ep_{EPOCHS}.json")

perceptron.predict(x_test, y_test, save_result=False)

pass
