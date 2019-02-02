import json

import numpy as np

from models.rbm import RBM
from util import sigmoid, sample_bernoulli, sample_bernoulli_batch


class FRBM(RBM):
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False):
        super().__init__(
            n_visible, n_hidden,
            learning_rate=learning_rate, momentum=momentum,
            xavier_const=xavier_const, err_function=err_function,
            use_tqdm=use_tqdm)

        # v to h links are w

        # self.w_l = xavier_init(self.n_visible, self.n_hidden, const=xavier_const-0.5)
        # self.w_r = xavier_init(self.n_visible, self.n_hidden, const=xavier_const+0.5)

        self.w_l = np.random.uniform(-0.1, 0.1, size=(self.n_visible, self.n_hidden))
        self.w_r = self.w_l + np.random.uniform(0.01, 0.1, size=(self.n_visible, self.n_hidden))

        # Visible bias = bv

        self.bv_l = np.random.uniform(-0.1, 0.1, size=self.n_visible)
        self.bv_r = self.bv_l + np.random.uniform(0.01, 0.1, size=self.n_visible)

        # Hidden bias = bh

        self.bh_l = np.random.uniform(-0.1, 0.1, size=self.n_hidden)
        self.bh_r = self.bh_l + np.random.uniform(0.01, 0.1, size=self.n_hidden)

    def get_prob_h(self, x_l, x_r):
        return (
            sigmoid(np.dot(np.transpose(self.w_l), x_l) + self.bh_l),
            sigmoid(np.dot(np.transpose(self.w_r), x_r) + self.bh_r)
        )

    def get_prob_h_batch(self, xs_l, xs_r):
        return (
            sigmoid(np.matmul(self.w_l.T, xs_l.T).T + self.bh_l),
            sigmoid(np.matmul(self.w_r.T, xs_r.T).T + self.bh_r)
        )

    def sample_h(self, ph_l, ph_r):
        return (
            sample_bernoulli(ph_l),
            sample_bernoulli(ph_r),
        )

    def sample_h_batch(self, ph_l, ph_r):
        return (
            sample_bernoulli_batch(ph_l),
            sample_bernoulli_batch(ph_r),
        )

    def get_prob_v(self, h_l, h_r):
        return (
            sigmoid(np.dot(self.w_l, h_l) + self.bv_l),
            sigmoid(np.dot(self.w_r, h_r) + self.bv_r)
        )

    def get_prob_v_batch(self, h_l, h_r):
        return (
            sigmoid(np.matmul(self.w_l, h_l.T).T + self.bv_l),
            sigmoid(np.matmul(self.w_r, h_r.T).T + self.bv_r)
        )

    def sample_v(self, pv_l, pv_r):
        return (
            sample_bernoulli(pv_l),
            sample_bernoulli(pv_r)
        )

    def sample_v_batch(self, pv_l, pv_r):
        return (
            sample_bernoulli_batch(pv_l),
            sample_bernoulli_batch(pv_r)
        )

    def get_delta_w(self, v0, prob_h0, sample_v1, prob_h1):
        return ((np.dot(v0.reshape(-1, 1), np.transpose(prob_h0.reshape(-1, 1))) -
                 np.dot(sample_v1.reshape(-1, 1), np.transpose(prob_h1.reshape(-1, 1))))
                * (self.momentum * self.learning_rate))

    def get_delta_bv(self, v0, sample_v1):
        return (v0 - sample_v1) * (self.momentum * self.learning_rate)

    def get_delta_bh(self, prob_h0, prob_h1):
        return (prob_h0 - prob_h1) * (self.momentum * self.learning_rate)

    def partial_fit(self, batch_x):
        prob_h0_l, prob_h0_r = self.get_prob_h_batch(batch_x, batch_x)
        sample_h0_l, sample_h0_r = self.sample_h_batch(prob_h0_l, prob_h0_r)

        prob_v1_l, prob_v1_r = self.get_prob_v_batch(sample_h0_l, sample_h0_r)
        sample_v1_l, sample_v1_r = self.sample_v_batch(prob_v1_l, prob_v1_r)

        sample_v1_defuz = (sample_v1_l + sample_v1_r) / 2

        prob_h1_l, prob_h1_r = self.get_prob_h_batch(sample_v1_l, sample_v1_r)

        x_to_h0_l = np.matmul(batch_x.T, prob_h0_l) / batch_x.shape[0]
        x_to_h0_r = np.matmul(batch_x.T, prob_h0_r) / batch_x.shape[0]

        v1_to_h1_l = np.matmul(sample_v1_l.T, prob_h1_l) / batch_x.shape[0]
        v1_to_h1_r = np.matmul(sample_v1_r.T, prob_h1_r) / batch_x.shape[0]

        x_m_v1_l = batch_x - sample_v1_l
        x_m_v1_r = batch_x - sample_v1_r

        h0_m_h1_l = prob_h0_l - prob_h1_l
        h0_m_h1_r = prob_h0_r - prob_h1_r

        deltas_w_l = (x_to_h0_l - v1_to_h1_l) * self.momentum * self.learning_rate
        deltas_w_r = (x_to_h0_r - v1_to_h1_r) * self.momentum * self.learning_rate

        deltas_bv_l = x_m_v1_l * self.momentum * self.learning_rate
        deltas_bv_r = x_m_v1_r * self.momentum * self.learning_rate

        deltas_bh_l = h0_m_h1_l * self.momentum * self.learning_rate
        deltas_bh_r = h0_m_h1_r * self.momentum * self.learning_rate

        # Process deltas

        delta_bh_mean_l = np.mean(deltas_bh_l, axis=0)
        delta_bh_mean_r = np.mean(deltas_bh_r, axis=0)

        delta_bv_mean_l = np.mean(deltas_bv_l, axis=0)
        delta_bv_mean_r = np.mean(deltas_bv_r, axis=0)

        delta_w_mean_l = deltas_w_l
        delta_w_mean_r = deltas_w_r

        self.w_l = self.w_l + delta_w_mean_l
        self.w_r = self.w_r + delta_w_mean_r

        self.bh_l = self.bh_l + delta_bh_mean_l
        self.bh_r = self.bh_r + delta_bh_mean_r

        self.bv_l = self.bv_l + delta_bv_mean_l
        self.bv_r = self.bv_r + delta_bv_mean_r

        return sample_v1_defuz

    def reconstruct(self, x):
        prob_h0_l, prob_h0_r = self.get_prob_h(x, x)
        # sample_h0_l, sample_h0_r = self.sample_h(prob_h0_l, prob_h0_r)
        prob_v1_l, prob_v1_r = self.get_prob_v(prob_h0_l, prob_h0_r)
        # sample_v1_l, sample_v1_r = self.sample_v(prob_v1_l, prob_v1_r)
        sample_v1_l, sample_v1_r = prob_v1_l, prob_v1_r
        return (sample_v1_l + sample_v1_r) / 2

    def save_weights(self, filename):
        data = {
            'w_l': self.w_l.tolist(),
            'w_r': self.w_r.tolist(),
            'bh_l': self.bh_l.tolist(),
            'bh_r': self.bh_r.tolist(),
            'bv_l': self.bv_l.tolist(),
            'bv_r': self.bv_r.tolist(),
        }
        with open(filename, "w") as f:
            f.write(json.dumps(data))

    def load_weights(self, filename):
        with open(filename, "rb") as f:
            data = json.loads(f.read())

        self.w_l = np.array(data["w_l"])
        self.w_r = np.array(data["w_r"])

        self.bh_l = np.array(data["bh_l"])
        self.bh_r = np.array(data["bh_r"])

        self.bv_l = np.array(data["bv_l"])
        self.bv_r = np.array(data["bv_r"])

    def defuzzify_weights(self, filename):

        w = (self.w_l + self.w_r) / 2
        bh = (self.bh_l + self.bh_r) / 2
        bv = (self.bv_l + self.bv_r) / 2

        data = {
            'w': w.tolist(),
            'bh': bh.tolist(),
            'bv': bv.tolist(),
        }
        with open(filename, "w") as f:
            f.write(json.dumps(data))
