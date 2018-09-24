import json
import numpy as np

from models.rbm import CRBM, FRBM


class DBN:
    RBM_MODEL = None

    def __init__(self, layers, learning_rate=0.01, momentum=0.95, err_function='mse', use_tqdm=False, **kwargs):

        self.rbms = [
            (self.RBM_MODEL(layers[i], layers[i + 1], use_tqdm=True, learning_rate=0.01))
            for i in range(0, len(layers) - 1)
        ]

    def prop_signal_forward(self, Xs, prop_to=0):
        raise NotImplementedError()

    def train_rbms(self, Xs, epochs=10, train_from=0, save_result=False):
        """
        train RMBs
        """

        Xss = self.prop_signal_forward(Xs, train_from or 0)

        print("Start Training RBMs")

        for i in range(train_from, len(self.rbms)):
            rbm = self.rbms[i]
            print(f"Current {rbm.__class__.__name__.upper()}: {rbm.n_visible}x{rbm.n_hidden}")
            rbm.fit(Xss, n_epoches=epochs, batch_size=10, shuffle=True, verbose=True, save_result=save_result)
            Xss = self.prop_signal_forward(Xs, i + 1)
            pass

        # self.rbms_to_dbn_weights()

    def fine_train(self, Xs, Ys):
        """
        fine tune the net
        """
        pass

    def rbms_to_dbn_weights(self):
        for rbm in self.rbms:
            self.weights.append(np.vstack((rbm.w, rbm.bh)).transpose())

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
        self.weights = [np.array(w) for w in data["weights"]]

    def reconstruct(self, x):
        signal = x
        for i, r in enumerate(self.rbms):
            prob_h0 = r.get_prob_h(signal)
            # signal = r.sample_h(prob_h0)
            signal = prob_h0

        sample = signal
        for i, r in enumerate(reversed(self.rbms)):
            prob_v1 = r.get_prob_v(sample)
            # sample = r.sample_v(prob_v1)
            sample = prob_v1

        return sample


class CrispDBN(DBN):
    RBM_MODEL = CRBM

    def prop_signal_forward(self, Xs, prop_to=0):
        signal = Xs
        if prop_to:
            for i, r in enumerate(self.rbms[:prop_to]):
                prob_h0 = r.get_prob_h_batch(signal)
                # signal = r.sample_h(prob_h0)
                signal = prob_h0
        return signal

    def reconstruct(self, x):
        signal = x
        for i, r in enumerate(self.rbms):
            prob_h0 = r.get_prob_h(signal)
            # signal = r.sample_h(prob_h0)
            signal = prob_h0

        sample = signal
        for i, r in enumerate(reversed(self.rbms)):
            prob_v1 = r.get_prob_v(sample)
            # sample = r.sample_v(prob_v1)
            sample = prob_v1

        return sample


class FuzzyDBN(DBN):
    RBM_MODEL = FRBM

    def prop_signal_forward(self, Xs, prop_to=0):
        signal = (Xs, Xs)
        if prop_to:
            for i, r in enumerate(self.rbms[:prop_to]):
                prob_h0_l, prob_h0_r = r.get_prob_h_batch(*signal)
                # signal_l, signal_r = r.sample_h(prob_h0_l, prob_h0_r)
                # signal = (signal_l + signal_r) / 2
                signal = (prob_h0_l, prob_h0_r)
        return signal

    def reconstruct(self, x):
        signal = x

        for i, r in enumerate(self.rbms):
            prob_h0_l, prob_h0_r = r.get_prob_h_batch(signal, signal)
            # signal_l, signal_r = r.sample_h(prob_h0_l, prob_h0_r)
            # signal = (signal_l + signal_r) / 2
            signal = (prob_h0_l + prob_h0_r) / 2

        for i, r in enumerate(reversed(self.rbms)):
            prob_v1_l, prob_v1_r = r.get_prob_v_batch(signal, signal)
            # sample = r.sample_v(prob_v1)
            signal = (prob_v1_l + prob_v1_r) / 2

        return signal
