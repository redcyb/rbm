import json

import numpy as np

from common import AbstractRBM
from util import xavier_init, sigmoid, sample_bernoulli


class RBM(AbstractRBM):
    def __init__(self, n_visible, n_hidden,
                 learning_rate=0.01, momentum=0.95, xavier_const=1.0,
                 err_function='mse', use_tqdm=False, seed=12345):

        super().__init__(
            n_visible, n_hidden,
            learning_rate=learning_rate, momentum=momentum, xavier_const=xavier_const, err_function=err_function,
            use_tqdm=use_tqdm, seed=seed)

        # v to h links are w

        # self.w = xavier_init(self.n_visible, self.n_hidden, const=xavier_const)
        self.w = np.random.uniform(-0.2, 0.2, size=(self.n_visible, self.n_hidden))
        self.shift_w = np.random.uniform(0, 0.2, size=(self.n_visible, self.n_hidden))

        # Visible bias = bv
        self.bv = np.random.uniform(-0.2, 0.2, size=self.n_visible)
        self.shift_bv = np.random.uniform(0, 0.2, size=self.n_visible)

        # Hidden bias = bh
        self.bh = np.random.uniform(-0.2, 0.2, size=self.n_hidden)
        self.shift_bh = np.random.uniform(0, 0.2, size=self.n_hidden)

    def get_prob_h(self, x):
        return sigmoid(np.dot(np.transpose(self.w), x) + self.bh)

    def sample_h(self, ph):
        return sample_bernoulli(ph)

    def get_prob_v(self, h):
        return sigmoid(np.dot(self.w, h) + self.bv)

    def sample_v(self, pv):
        return sample_bernoulli(pv)

    def reconstruct(self, x):
        prob_h0 = self.get_prob_h(x)
        # sample_h0 = self.sample_h(prob_h0)
        prob_v1 = self.get_prob_v(prob_h0)
        # sample_v1 = self.sample_v(prob_v1)
        sample_v1 = prob_v1
        return sample_v1

    def partial_fit(self, batch_x):

        deltas_w = []
        deltas_bh = []
        deltas_bv = []

        batch_sample_v1 = []

        for x0 in batch_x:
            prob_h0 = self.get_prob_h(x0)
            sample_h0 = self.sample_h(prob_h0)

            prob_v1 = self.get_prob_v(sample_h0)
            sample_v1 = self.sample_v(prob_v1)

            prob_h1 = self.get_prob_h(sample_v1)

            delta_w = ((np.dot(x0.reshape(-1, 1), np.transpose(prob_h0.reshape(-1, 1))) -
                        np.dot(sample_v1.reshape(-1, 1), np.transpose(prob_h1.reshape(-1, 1))))
                       * (self.momentum * self.learning_rate))
            delta_vb = ((x0 - sample_v1) * (self.momentum * self.learning_rate))
            delta_hb = ((prob_h0 - prob_h1) * (self.momentum * self.learning_rate))

            deltas_w.append(delta_w)
            deltas_bh.append(delta_hb)
            deltas_bv.append(delta_vb)

            batch_sample_v1.append(sample_v1)

        deltas_bh = np.array(deltas_bh)
        deltas_bv = np.array(deltas_bv)
        deltas_w = np.array(deltas_w)

        delta_bh_mean = np.mean(deltas_bh, axis=0)
        delta_bv_mean = np.mean(deltas_bv, axis=0)
        delta_w_mean = np.mean(deltas_w, axis=0)

        self.w = self.w + delta_w_mean
        self.bh = self.bh + delta_bh_mean
        self.bv = self.bv + delta_bv_mean

        return np.array(batch_sample_v1)

    def save_weights(self, filename):
        data = {
            'w': self.w.tolist(),
            'shift_w': self.shift_w.tolist(),
            'bh': self.bh.tolist(),
            'shift_bh': self.shift_bh.tolist(),
            'bv': self.bv.tolist(),
            'shift_bv': self.shift_bv.tolist(),
        }
        d = json.dumps(data)

        with open(filename, "w") as f:
            f.write(d)

    def save_details(self, filename, data):
        d = json.dumps(data)
        with open(filename, "w") as f:
            f.write(d)

    def load_weights(self, filename):
        with open(filename, "rb") as f:
            data = json.loads(f.read())
        self.w = np.array(data["w"])
        self.shift_w = np.array(data["shift_w"])
        self.bh = np.array(data["bh"])
        self.shift_bh = np.array(data["shift_bh"])
        self.bv = np.array(data["bv"])
        self.shift_bv = np.array(data["shift_bv"])
