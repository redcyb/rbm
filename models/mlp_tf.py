import json
import sys

import numpy as np

from util import save_original_digit_image, save_digit_image, label_to_str
from .functions import activate, derivative, derivative_batch


class Perceptron:
    def __init__(self, layers, **kwargs):

        """
        Properties:

        layers_A = Activated Output of each layer: layers_A[l][i] = activation_func(layers_Z[l][i])
        layers_Z = Weighted Sum of inputs of each layer: sum(xi * wji)

        self.AOs = Array of predicted Outputs for whole dataset

        weights = weights between all Layers
        weights_deltas = weights deltas to update weights after each dataset example handling

        self.EO = 1/2 * SUM(Ysn - self.AO)^2 for 1 Dataset Example (online). I.e. errors by output neurons
        self.AO_m_Y = dEO / dAO   where AO means layers_A[-1] aka last layer or output layer

        self.ETotal = sum(self.EO) for 1 Dataset Example (online).

        """

        inputs_size = layers[0]
        outputs_size = layers[-1]
        hidden_shape = [i for i in layers[1:-1]]

        self.layers_number = len(layers)

        self.layers_A = [np.array([None] * i) for i in layers]
        self.layers_Z = [np.array([None] * i) for i in layers]

        self.w = [np.random.normal(0, 0.05, (len(self.layers_A[i]), len(self.layers_A[i - 1]))) for i in
                  range(1, self.layers_number)]

        self.w_deltas = [np.zeros((len(self.layers_Z[i]), len(self.layers_A[i - 1])))
                         for i in range(1, self.layers_number)]

        self.b = [np.random.normal(0, 0.05, len(self.layers_A[i])) for i in range(1, self.layers_number)]
        self.b_deltas = [np.zeros(len(self.layers_A[i])) for i in range(1, self.layers_number)]

        self.AO_m_Y = []
        self.EO = []
        self.ETotal = []

        self.AOs = []

    def re_init(self, weights):
        pass

    def forward(self, Xsn, Ysn):
        self.layers_A[0] = Xsn
        for i in range(1, self.layers_number):
            w = self.w[i - 1]
            x = self.layers_A[i - 1]
            b = self.b[i - 1]
            self.layers_Z[i] = np.matmul(w, x).T + b
            self.layers_A[i] = activate("sigmoid", self.layers_Z[i])
            pass

        self.AO_m_Y = self.layers_A[-1] - Ysn
        self.EO = np.array(
            np.power((Ysn - self.layers_A[-1]), np.array([2 for i in range(len(self.layers_A[-1]))])) / 2)
        self.ETotal = np.sum(self.EO)

        pass

    def backward(self, learning_rate=0.01):
        # Layer H-O

        dE_by_dO = self.AO_m_Y  # dEi/dOut for squared sum loss function
        dAO_by_ZO = derivative("sigmoid", self.layers_Z[-1])
        dZO_by_HOW = self.layers_A[-2]
        grads = dE_by_dO * dAO_by_ZO
        dE_by_HOW = np.array([[grads[i] * dZO_by_HOW[j] for j in range(len(dZO_by_HOW))] for i in range(len(grads))])
        self.w_deltas[-1] = learning_rate * dE_by_HOW

        for k in range(len(self.layers_A) - 2, 0, -1):
            # n means next layer

            dAk_by_Zk = derivative("sigmoid", self.layers_Z[k])
            dZk_by_Wk = self.layers_A[k - 1]  # just activated outputs of previous layer

            dEn_by_Zn = grads
            dZn_by_Ak = self.w[k][:, :-1]  # we don't need bias in HO layer for AH

            dE_by_Ak = np.array(np.dot(np.matrix(dZn_by_Ak).T, dEn_by_Zn))[0]

            grads = dE_by_Ak * dAk_by_Zk

            dE_by_Wk = np.array([[grads[i] * dZk_by_Wk[j] for j in range(len(dZk_by_Wk))] for i in range(len(grads))])
            self.w_deltas[k - 1] = learning_rate * dE_by_Wk

        # update all weights

        for i in range(len(self.w)):
            self.w[i] -= self.w_deltas[i]

    def forward_batch(self, Xs, Ys):
        self.layers_A[0] = Xs
        for i in range(1, self.layers_number):
            w = self.w[i - 1]
            x = self.layers_A[i - 1].T
            b = self.b[i - 1]
            self.layers_Z[i] = np.matmul(w, x).T + b
            self.layers_A[i] = activate("sigmoid", self.layers_Z[i])

        twos = np.ones(self.layers_A[-1].shape) * 2

        self.AO_m_Y = np.mean(self.layers_A[-1] - Ys, axis=0, dtype=np.float32)
        self.EO = np.array(np.power(self.AO_m_Y, twos) / 2, dtype=np.float32)
        self.ETotal = np.sum(self.EO, dtype=np.float32)

    def backward_batch(self, learning_rate=0.01, only_last=False):
        # Layer H-O

        dE_by_dO = self.AO_m_Y  # dEi/dOut for squared sum loss function

        dAO_by_ZO = np.mean(derivative_batch("sigmoid", self.layers_Z[-1]), axis=0, dtype=np.float32)
        dZO_by_HOW = self.layers_A[-2]

        grads = dE_by_dO * dAO_by_ZO

        dE_by_HOW = np.array([dZO_by_HOW * g for g in grads])

        self.w_deltas[-1] = np.mean(learning_rate * dE_by_HOW, axis=1)
        self.b_deltas[-1] = np.mean(learning_rate * grads)

        if not only_last:

            for k in range(self.layers_number - 2, 0, -1):
                # n means next layer

                dAk_by_Zk = np.mean(derivative_batch("sigmoid", self.layers_Z[k]), axis=0)
                dZk_by_Wk = self.layers_A[k - 1]  # just activated outputs of previous layer

                dEn_by_Zn = grads
                dZn_by_Ak = self.w[k]  # we don't need bias in HO layer for AH

                dE_by_Ak = np.array(np.dot(np.matrix(dZn_by_Ak).T, dEn_by_Zn))[0]

                grads = dE_by_Ak * dAk_by_Zk

                dE_by_Wk = np.array([dZk_by_Wk * g for g in grads])
                dE_by_Bk = grads

                self.w_deltas[k - 1] = np.mean(learning_rate * dE_by_Wk, axis=1)
                self.b_deltas[k - 1] = np.mean(learning_rate * dE_by_Bk)

                pass

        # update all weights

        for i in range(len(self.w)):
            self.w[i] -= self.w_deltas[i]

        for i in range(len(self.b)):
            self.b[i] -= self.b_deltas[i]

    def train_online(self, Xs, Ys, epochs=1000, learning_rate=0.1, threshold=0.01, report_every=100, min_epochs=1):
        epoch = 0

        for i in range(epochs):
            epoch += 1

            for n in range(Xs.shape[0]):
                self.forward(Xs[n], Ys[n])
                self.backward(learning_rate)

            if self.ETotal < threshold and epoch > min_epochs:
                break

            if report_every and not epoch % report_every:
                print(f"\nEpoch: {epoch}")
                print(f"Error: {self.ETotal}")

        print(f"\nTraining epochs: {epoch}")
        print(f"Training Error: {self.ETotal}\n")

    def train_batch(self, Xs, Ys, epochs=1000, learning_rate=0.01, threshold=0.01, report_every=100, min_epochs=1,
                    batch_size=10, only_last=False):

        from tqdm import tqdm

        epoch = 0

        n_data = Xs.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        r_batches = range(n_batches)

        for i in range(epochs):
            epoch += 1

            r_batches = tqdm(r_batches, desc='Epoch: {:d}'.format(i), ascii=True, file=sys.stdout)

            for b in r_batches:
                Xss = Xs[b * batch_size:(b + 1) * batch_size]
                Yss = Ys[b * batch_size:(b + 1) * batch_size]

                self.forward_batch(Xss, Yss)
                self.backward_batch(learning_rate, only_last=only_last)

            if self.ETotal < threshold and epoch > min_epochs:
                break

            err_mean = self.ETotal

            tqdm.write('Train error: {:.6f}'.format(err_mean))
            tqdm.write('')
            sys.stdout.flush()

            # if report_every and not epoch % report_every:
            #     print(f"\nEpoch: {epoch}")
            #     print(f"Error: {self.ETotal}")

        # print(f"\nTraining epochs: {epoch}")
        # print(f"Training Error: {self.ETotal}\n")

    def predict(self, Xs, Ys, save_result=False):
        print("Predicted:\n")

        count = 0

        for n in range(Xs.shape[0]):
            self.forward(Xs[n], Ys[n])
            # self.AOs.append(self.layers_A[-1])
            # print(f"X: {Xs[n]}    O: {[f'{i:.2f}' for i in self.layers_A[-1]]}    Y: {[f'{i:.2f}' for i in Ys[n]]}")

            origin = np.argmax(Ys[n])
            reprod = np.argmax(self.layers_A[-1])

            count += 1 if origin == reprod else 0

            # print(f"O: {reprod}    Y: {origin}    Matched?: {origin == reprod}")

            # if save_result:
            #     save_digit_image(Xs[n], "n", label_to_str(Ys[n]), f"{origin==reprod}___{origin}_vs_{reprod}___MLP.png")

        # self.AOs = np.array(self.AOs)

        print(f"\nTest Error: {self.ETotal}\n")
        print(f"\nMatch: {count}   NotMatch: {Xs.shape[0] - count}   from {Xs.shape[0]}")

    def save_weights(self, filename):
        data = {
            "w": [w.tolist() for w in self.w],
            "b": [b.tolist() for b in self.b]
        }
        d = json.dumps(data)

        with open(filename, "w") as f:
            f.write(d)

    def load_weights(self, filename):
        with open(filename, "rb") as f:
            data = json.loads(f.read())
        self.w = [np.array(w) for w in data["w"]]
        self.b = [np.array(b) for b in data["b"]]
