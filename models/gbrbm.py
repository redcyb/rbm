import json

import numpy as np

from models.rbm import RBM
from util import sigmoid, sample_bernoulli, sample_bernoulli_batch


class GBRBM(RBM):
    def __init__(self, n_visible, n_hidden,
                 learning_rate=0.01, momentum=1, xavier_const=1.0,
                 err_function='mse', use_tqdm=False):
        super().__init__(
            n_visible, n_hidden,
            learning_rate=learning_rate, momentum=momentum, xavier_const=xavier_const, err_function=err_function,
            use_tqdm=use_tqdm)

        # v to h links are w

        # self.w = xavier_init(self.n_visible, self.n_hidden, const=xavier_const)
        self.w = np.random.uniform(-0.2, 0.2, size=(self.n_visible, self.n_hidden))
        self.shift_w = np.random.uniform(0, 0.2, size=(self.n_visible, self.n_hidden))

        # Visible bias = bv
        self.bv = np.random.uniform(-0.2, 0.2, size=self.n_visible)
        self.shift_bv = np.random.uniform(0, 0.2, size=self.n_visible)

        # Hidden bias = bh
        # self.bh = np.ones(self.n_hidden)
        self.bh = np.random.uniform(-0.2, 0.2, size=self.n_hidden)
        self.shift_bh = np.random.uniform(0, 0.2, size=self.n_hidden)

    def get_prob_h(self, x):
        return sigmoid(np.dot(np.transpose(self.w), x) + self.bh)

    def get_prob_h_batch(self, xs):
        return sigmoid(np.matmul(np.transpose(self.w), xs.transpose()).T + self.bh)

    def sample_h(self, ph):
        return sample_bernoulli(ph)

    def sample_h_batch(self, ph):
        return sample_bernoulli_batch(ph)

    def get_prob_v(self, h):
        return sigmoid(np.dot(self.w, h) + self.bv)

    def get_prob_v_batch(self, hs):
        return sigmoid(np.matmul(self.w, hs.T).T + self.bv)

    def sample_v(self, pv):
        return sample_bernoulli(pv)

    def sample_v_batch(self, pv):
        return sample_bernoulli_batch(pv)

    def reconstruct(self, x):
        prob_h0 = self.get_prob_h(x)
        # sample_h0 = self.sample_h(prob_h0)
        prob_v1 = self.get_prob_v(prob_h0)
        # sample_v1 = self.sample_v(prob_v1)
        sample_v1 = prob_v1
        return sample_v1

    def partial_fit(self, batch_x):
        prob_h0 = self.get_prob_h_batch(batch_x)
        sample_h0 = self.sample_h_batch(prob_h0)

        prob_v1 = self.get_prob_v_batch(sample_h0)
        sample_v1 = self.sample_v_batch(prob_v1)

        prob_h1 = self.get_prob_h_batch(sample_v1)

        x_to_h0 = np.matmul(batch_x.T, prob_h0) / batch_x.shape[0]
        v1_to_h1 = np.matmul(sample_v1.T, prob_h1) / batch_x.shape[0]
        x_m_v1 = batch_x - sample_v1
        h0_m_h1 = prob_h0 - prob_h1

        deltas_w = (x_to_h0 - v1_to_h1) * self.momentum * self.learning_rate
        deltas_bv = x_m_v1 * self.momentum * self.learning_rate
        deltas_bh = h0_m_h1 * self.momentum * self.learning_rate

        # Process deltas

        delta_bh_mean = np.mean(deltas_bh, axis=0)
        delta_bv_mean = np.mean(deltas_bv, axis=0)
        delta_w_mean = deltas_w

        self.w = self.w + delta_w_mean
        self.bh = self.bh + delta_bh_mean
        self.bv = self.bv + delta_bv_mean

        return sample_v1

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
        # self.shift_w = np.array(data["shift_w"])
        self.bh = np.array(data["bh"])
        # self.shift_bh = np.array(data["shift_bh"])
        self.bv = np.array(data["bv"])
        # self.shift_bv = np.array(data["shift_bv"])
