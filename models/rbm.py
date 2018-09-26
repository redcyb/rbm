import json

import numpy as np

from util import sigmoid, sample_bernoulli, sample_bernoulli_batch

import json
from datetime import datetime

import sys


class RBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False,
                 **kwargs):

        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function != 'mse':
            raise ValueError('err_function should be \'mse\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        # v to h links are w

        # Visible bias = bv

        # Hidden bias = bh

    def get_err(self, batch_x, batch_sample_v1, err_func="mse"):
        # return np.mean(np.square(np.power((batch_x - batch_sample_v1), 2 * np.ones(batch_x.shape))))
        return np.mean(np.square((batch_x - batch_sample_v1)))

    def get_free_energy(self):
        raise NotImplementedError()

    def reconstruct(self, x):
        raise NotImplementedError()

    def partial_fit(self, batch_x):
        raise NotImplementedError()

    def fit(self, data_x, n_epoches=10, batch_size=10, shuffle=True, verbose=True, save_result=False):

        # momentum_step = self.momentum / n_epoches / 1.5

        start = datetime.now()

        assert n_epoches > 0

        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        inds = np.arange(n_data)

        if shuffle:
            data_x_cpy = data_x.copy()
        else:
            data_x_cpy = data_x

        all_errs = []
        epochs_mean_errs = []

        for e in range(n_epoches):
            # self.momentum -= momentum_step
            # print(f"lr: {self.learning_rate*self.momentum}")

            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))

            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                batch_sample_v1 = self.partial_fit(batch_x)

                batch_err = self.get_err(batch_x, batch_sample_v1)

                epoch_errs[epoch_errs_ptr] = batch_err

                epoch_errs_ptr += 1

            if verbose:

                err_mean = epoch_errs.mean()
                epochs_mean_errs.append(err_mean)

                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.6f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.6f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            all_errs = np.hstack([all_errs, epoch_errs])

        if save_result:
            self.save_weights(
                f"./weights/{self.__class__.__name__.lower()}___{self.n_visible}x{self.n_hidden}___ep_{n_epoches}.json"
            )
            self.save_details(
                f"./details/{self.__class__.__name__.lower()}___{self.n_visible}x{self.n_hidden}___ep_{n_epoches}.json",
                {"training_time (s)": (datetime.now() - start).seconds, "errors": epochs_mean_errs}
            )

        return all_errs, epochs_mean_errs

    def save_weights(self, filename):
        raise NotImplementedError()

    def save_details(self, filename, data):
        d = json.dumps(data)
        with open(filename, "w") as f:
            f.write(d)

    def load_weights(self, filename):
        raise NotImplementedError()


class CRBM(RBM):
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
        self.shift_w = np.array(data["shift_w"])
        self.bh = np.array(data["bh"])
        self.shift_bh = np.array(data["shift_bh"])
        self.bv = np.array(data["bv"])
        self.shift_bv = np.array(data["shift_bv"])


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

    def fit(self, xs, n_epoches=10, batch_size=10, shuffle=True, verbose=True, save_result=False):
        start = datetime.now()

        if isinstance(xs, (tuple, list)):
            x_l = xs[0]
            x_r = xs[1]
        else:
            x_l = xs
            x_r = xs

        n_data = x_l.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        inds = np.arange(n_data)

        if shuffle:
            x_l_cpy = x_l.copy()
            x_r_cpy = x_r.copy()
        else:
            x_l_cpy = x_l
            x_r_cpy = x_r

        all_errs = []
        epochs_mean_errs = []

        for e in range(n_epoches):
            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = []

            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)

                x_l_cpy = x_l_cpy[inds]
                x_r_cpy = x_r_cpy[inds]

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x_l = x_l_cpy[b * batch_size:(b + 1) * batch_size]
                batch_x_r = x_r_cpy[b * batch_size:(b + 1) * batch_size]

                batch_sample_v1 = self.partial_fit((batch_x_l, batch_x_r))

                batch_err = self.get_err((batch_x_l, batch_x_r), batch_sample_v1)

                epoch_errs.append(batch_err)

                epoch_errs_ptr += 1

            if verbose:

                epoch_errs_m = np.array(epoch_errs)

                err_mean = epoch_errs_m.mean(axis=0)
                epochs_mean_errs.append(err_mean)

                if self._use_tqdm:
                    self._tqdm.write(f'Train error: {err_mean}')
                    self._tqdm.write('')
                else:
                    print(f'Train error: {err_mean}')
                    print('')
                sys.stdout.flush()

            # all_errs = np.hstack([all_errs, epoch_errs])

        if save_result:
            self.save_weights(
                f"./weights/{self.__class__.__name__.lower()}___{self.n_visible}x{self.n_hidden}___ep_{n_epoches}.json"
            )
            # self.save_details(
            #     f"./details/{self.__class__.__name__.lower()}___{self.n_visible}x{self.n_hidden}___ep_{n_epoches}.json",
            #     {"training_time (s)": (datetime.now() - start).seconds, "errors": epochs_mean_errs}
            # )

        return all_errs, epochs_mean_errs

    def get_err(self, batch_x, batch_sample_v1, err_func="mse"):
        x_l, x_r = batch_x
        v_l, v_r = batch_sample_v1
        return np.mean(np.square(x_l - v_l)), np.mean(np.square(x_r - v_r))

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

    def partial_fit(self, batch_x_pair):
        x_l, x_r = batch_x_pair

        prob_h0_l, prob_h0_r = self.get_prob_h_batch(x_l, x_r)
        sample_h0_l, sample_h0_r = self.sample_h_batch(prob_h0_l, prob_h0_r)

        prob_v1_l, prob_v1_r = self.get_prob_v_batch(sample_h0_l, sample_h0_r)
        sample_v1_l, sample_v1_r = self.sample_v_batch(prob_v1_l, prob_v1_r)

        prob_h1_l, prob_h1_r = self.get_prob_h_batch(sample_v1_l, sample_v1_r)

        x_to_h0_l = np.matmul(x_l.T, prob_h0_l) / x_l.shape[0]
        x_to_h0_r = np.matmul(x_r.T, prob_h0_r) / x_r.shape[0]

        v1_to_h1_l = np.matmul(sample_v1_l.T, prob_h1_l) / x_l.shape[0]
        v1_to_h1_r = np.matmul(sample_v1_r.T, prob_h1_r) / x_r.shape[0]

        x_m_v1_l = x_l - sample_v1_l
        x_m_v1_r = x_r - sample_v1_r

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

        return sample_v1_l, sample_v1_r

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
