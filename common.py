import json

import numpy as np
import sys


class AbstractRBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False):

        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

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
        return np.mean(np.square(np.power((batch_x - batch_sample_v1), 2 * np.ones(batch_x.shape))))

    def get_free_energy(self):
        raise NotImplementedError()

    def reconstruct(self, x):
        raise NotImplementedError()

    def partial_fit(self, batch_x):
        raise NotImplementedError()

    def fit(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            shuffle=True,
            verbose=True):
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
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            all_errs = np.hstack([all_errs, epoch_errs])

        return all_errs, epochs_mean_errs

    def save_weights(self, filename):
        raise NotImplementedError()

    def save_details(self, filename, data):
        d = json.dumps(data)
        with open(filename, "w") as f:
            f.write(d)

    def load_weights(self, filename):
        raise NotImplementedError()
