import json
import numpy as np

from common import AbstractRBM
from util import xavier_init, sigmoid, sample_bernoulli


class FRBM(AbstractRBM):
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

    def sample_h(self, ph_l, ph_r):
        return (
            sample_bernoulli(ph_l),
            sample_bernoulli(ph_r),
        )

    def get_prob_v(self, h_l, h_r):
        return (
            sigmoid(np.dot(self.w_l, h_l) + self.bv_l),
            sigmoid(np.dot(self.w_r, h_r) + self.bv_r)
        )

    def sample_v(self, pv_l, pv_r):
        return (
            sample_bernoulli(pv_l),
            sample_bernoulli(pv_r)
        )

    def partial_fit(self, batch_x):
        deltas_w_l, deltas_w_r, deltas_bh_l, deltas_bh_r, deltas_bv_l, deltas_bv_r = [[] for i in range(6)]
        batch_sample_v1 = []

        for x0 in batch_x:
            prob_h0_l, prob_h0_r = self.get_prob_h(x0, x0)
            sample_h0_l, sample_h0_r = self.sample_h(prob_h0_l, prob_h0_r)
            prob_v1_l, prob_v1_r = self.get_prob_v(sample_h0_l, sample_h0_r)
            sample_v1_l, sample_v1_r = self.sample_v(prob_v1_l, prob_v1_r)
            sample_v1_defuz = (sample_v1_l + sample_v1_r) / 2
            prob_h1_l, prob_h1_r = self.get_prob_h(sample_v1_l, sample_v1_r)

            # Deltas

            delta_w_l = ((np.dot(x0.reshape(-1, 1), np.transpose(prob_h0_l.reshape(-1, 1))) -
                          np.dot(sample_v1_l.reshape(-1, 1), np.transpose(prob_h1_l.reshape(-1, 1))))
                         * (self.momentum * self.learning_rate))
            delta_vb_l = ((x0 - sample_v1_l) * (self.momentum * self.learning_rate))
            delta_hb_l = ((prob_h0_l - prob_h1_l) * (self.momentum * self.learning_rate))

            delta_w_r = ((np.dot(x0.reshape(-1, 1), np.transpose(prob_h0_r.reshape(-1, 1))) -
                          np.dot(sample_v1_r.reshape(-1, 1), np.transpose(prob_h1_r.reshape(-1, 1))))
                         * (self.momentum * self.learning_rate))
            delta_vb_r = ((x0 - sample_v1_r) * (self.momentum * self.learning_rate))
            delta_hb_r = ((prob_h0_r - prob_h1_r) * (self.momentum * self.learning_rate))

            deltas_w_l.append(delta_w_l)
            deltas_w_r.append(delta_w_r)

            deltas_bh_l.append(delta_hb_l)
            deltas_bh_r.append(delta_hb_r)

            deltas_bv_l.append(delta_vb_l)
            deltas_bv_r.append(delta_vb_r)

            # Batch result

            batch_sample_v1.append(sample_v1_defuz)

        deltas_bh_l = np.array(deltas_bh_l)
        deltas_bh_r = np.array(deltas_bh_r)
        deltas_bv_l = np.array(deltas_bv_l)
        deltas_bv_r = np.array(deltas_bv_r)
        deltas_w_l = np.array(deltas_w_l)
        deltas_w_r = np.array(deltas_w_r)

        delta_bh_mean_l = np.mean(deltas_bh_l, axis=0)
        delta_bh_mean_r = np.mean(deltas_bh_r, axis=0)
        delta_bv_mean_l = np.mean(deltas_bv_l, axis=0)
        delta_bv_mean_r = np.mean(deltas_bv_r, axis=0)
        delta_w_mean_l = np.mean(deltas_w_l, axis=0)
        delta_w_mean_r = np.mean(deltas_w_r, axis=0)

        self.w_l = self.w_l + delta_w_mean_l
        self.w_r = self.w_r + delta_w_mean_r
        self.bh_l = self.bh_l + delta_bh_mean_l
        self.bh_r = self.bh_r + delta_bh_mean_r
        self.bv_l = self.bv_l + delta_bv_mean_l
        self.bv_r = self.bv_r + delta_bv_mean_r

        return np.array(batch_sample_v1)

    def reconstruct(self, x):
        prob_h0_l, prob_h0_r = self.get_prob_h(x, x)
        sample_h0_l, sample_h0_r = self.sample_h(prob_h0_l, prob_h0_r)
        prob_v1_l, prob_v1_r = self.get_prob_v(sample_h0_l, sample_h0_r)
        sample_v1_l, sample_v1_r = self.sample_v(prob_v1_l, prob_v1_r)
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
