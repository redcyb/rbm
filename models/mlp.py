import json

import numpy as np

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

        self.layers_A = np.array(
            [[None] * (inputs_size + 1), *[[None] * (i + 1) for i in hidden_shape], [None] * outputs_size]
        )

        self.layers_Z = np.array(
            [[None] * inputs_size, *[[None] * i for i in hidden_shape], [None] * outputs_size]
        )

        self.weights = [np.random.normal(0, 0.2, (len(self.layers_Z[i]), len(self.layers_A[i - 1])))
                        for i in range(1, self.layers_number)]

        self.weights_deltas = [np.zeros((len(self.layers_Z[i]), len(self.layers_A[i - 1])))
                               for i in range(1, self.layers_number)]

        self.AO_m_Y = []
        self.EO = []
        self.ETotal = []

        self.AOs = []

    def re_init(self, weights):
        pass

    def forward(self, Xsn, Ysn):
        self.layers_A[0] = np.append(Xsn, 1)

        for i in range(1, self.layers_number):
            a = self.weights[i - 1]
            b = self.layers_A[i - 1]
            c = np.dot(a, b)

            self.layers_Z[i] = c

            if i < self.layers_number - 1:
                act = np.append(activate("sigmoid", self.layers_Z[i]), 1)
            else:
                act = activate("sigmoid", self.layers_Z[i])

            self.layers_A[i] = act
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
        self.weights_deltas[-1] = learning_rate * dE_by_HOW

        for k in range(len(self.layers_A) - 2, 0, -1):
            # n means next layer

            dAk_by_Zk = derivative("sigmoid", self.layers_Z[k])
            dZk_by_Wk = self.layers_A[k - 1]  # just activated outputs of previous layer

            dEn_by_Zn = grads
            dZn_by_Ak = self.weights[k][:, :-1]  # we don't need bias in HO layer for AH

            dE_by_Ak = np.array(np.dot(np.matrix(dZn_by_Ak).T, dEn_by_Zn))[0]

            grads = dE_by_Ak * dAk_by_Zk

            dE_by_Wk = np.array([[grads[i] * dZk_by_Wk[j] for j in range(len(dZk_by_Wk))] for i in range(len(grads))])
            self.weights_deltas[k - 1] = learning_rate * dE_by_Wk

        # update all weights

        for i in range(len(self.weights)):
            self.weights[i] -= self.weights_deltas[i]

    def forward_batch(self, Xs, Ys):

        ones = np.ones((Xs.shape[0], 1))
        Xss = np.hstack((Xs, ones))

        self.layers_A[0] = Xss

        for i in range(1, self.layers_number):
            a = self.weights[i - 1]
            b = self.layers_A[i - 1].transpose()
            c = np.matmul(a, b).transpose()

            self.layers_Z[i] = c

            if i < self.layers_number - 1:
                ones = np.ones((self.layers_Z[i].shape[0], 1))
                act = np.hstack((activate("sigmoid", self.layers_Z[i]), ones))
            else:
                act = activate("sigmoid", self.layers_Z[i])

            self.layers_A[i] = act
            pass

        out = self.layers_A[-1]
        # mean_output = np.mean(self.layers_A[-1], axis=0)

        twos = np.ones(self.layers_A[-1].shape) * 2

        self.AO_m_Y = np.mean(self.layers_A[-1] - Ys, axis=0)

        self.EO = np.array(np.power(self.AO_m_Y, twos) / 2)

        self.ETotal = np.sum(self.EO)

        pass

    def backward_batch(self, learning_rate=0.01):
        # Layer H-O

        dE_by_dO = self.AO_m_Y  # dEi/dOut for squared sum loss function
        dAO_by_ZO = np.mean(derivative_batch("sigmoid", self.layers_Z[-1]), axis=0)
        dZO_by_HOW = self.layers_A[-2]
        grads = dE_by_dO * dAO_by_ZO
        dE_by_HOW = np.array([[grads[i] * dZO_by_HOW[j] for j in range(len(dZO_by_HOW))] for i in range(len(grads))])
        self.weights_deltas[-1] = np.mean(learning_rate * dE_by_HOW, axis=1)

        for k in range(self.layers_number-2, 0, -1):
            # n means next layer

            dAk_by_Zk = np.mean(derivative_batch("sigmoid", self.layers_Z[k]), axis=0)
            dZk_by_Wk = self.layers_A[k - 1]  # just activated outputs of previous layer

            dEn_by_Zn = grads
            dZn_by_Ak = self.weights[k][:, :-1]  # we don't need bias in HO layer for AH

            dE_by_Ak = np.array(np.dot(np.matrix(dZn_by_Ak).T, dEn_by_Zn))[0]

            grads = dE_by_Ak * dAk_by_Zk

            dE_by_Wk = np.array([[grads[i] * dZk_by_Wk[j] for j in range(len(dZk_by_Wk))] for i in range(len(grads))])
            self.weights_deltas[k - 1] = np.mean(learning_rate * dE_by_Wk, axis=1)

            pass

        # update all weights

        for i in range(len(self.weights)):
            self.weights[i] -= self.weights_deltas[i]

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

    def train_batch(self, Xs, Ys, epochs=1000, learning_rate=0.1, threshold=0.01, report_every=100, min_epochs=1,
                    batch_size=100):
        epoch = 0

        batches_num = int(Xs.shape[0] / batch_size)

        for i in range(epochs):
            epoch += 1

            for b in range(batches_num):
                from_ = b * batch_size
                to_ = None if b == batches_num - 1 else (b + 1) * batch_size

                Xss = Xs[from_:to_]
                Yss = Ys[from_:to_]

                self.forward_batch(Xss, Yss)
                self.backward_batch(learning_rate)

            if self.ETotal < threshold and epoch > min_epochs:
                break

            if report_every and not epoch % report_every:
                print(f"\nEpoch: {epoch}")
                print(f"Error: {self.ETotal}")

        print(f"\nTraining epochs: {epoch}")
        print(f"Training Error: {self.ETotal}\n")

    def predict(self, Xs, Ys):
        print("Predicted:\n")

        for n in range(Xs.shape[0]):
            self.forward(Xs[n], Ys[n])
            self.AOs.append(self.layers_A[-1])
            # print(f"X: {Xs[n]}    O: {[f'{i:.2f}' for i in self.layers_A[-1]]}    Y: {[f'{i:.2f}' for i in Ys[n]]}")
            print(f"O: {[f'{i:.2f}' for i in self.layers_A[-1]]}    Y: {[f'{i:.2f}' for i in Ys[n]]}")

        self.AOs = np.array(self.AOs)

        print(f"\nTest Error: {self.ETotal}\n")

    def save_weights(self, filename):
        data = {
            "weights": [w.tolist() for w in self.weights]
        }
        d = json.dumps(data)

        with open(filename, "w") as f:
            f.write(d)

    def load_weights(self, filename):
        with open(filename, "rb") as f:
            data = json.loads(f.read())
        self.weights = [np.array(w) for w in data["weights"][:2]]
