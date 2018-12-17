import tensorflow as tf
import numpy as np
import sys

from .util import sample_bernoulli, tf_xavier_init


class FuzzyBerBerRBM:
    def __init__(self, n_visible, n_hidden,
                 learning_rate=0.01, momentum=0.95, xavier_const=1.0, err_function='mse', use_tqdm=False):

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

        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w_l = tf.Variable(tf.random_normal((self.n_visible, self.n_hidden), stddev=0.1),
                               dtype=tf.float32)
        self.w_r = tf.Variable(self.w_l + tf.random_normal((self.n_visible, self.n_hidden), stddev=0.1),
                               dtype=tf.float32)

        # Visible bias = b

        self.bv_l = tf.Variable(tf.random_normal([self.n_visible], stddev=0.1), dtype=tf.float32)
        self.bv_r = tf.Variable(self.bv_l + tf.random_normal([self.n_visible], stddev=0.1), dtype=tf.float32)

        # Hidden bias = c

        self.bh_l = tf.Variable(tf.random_normal([self.n_hidden], stddev=0.1), dtype=tf.float32)
        self.bh_r = tf.Variable(self.bh_l + tf.random_normal([self.n_hidden], stddev=0.1), dtype=tf.float32)

        self.delta_w_l = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_w_r = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)

        self.delta_bv_l = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_bv_r = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)

        self.delta_bh_l = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
        self.delta_bh_r = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        # Actions

        self.update_weights = None
        self.update_deltas = None

        self.sample_h0_l = None
        self.sample_h0_r = None

        self.sample_v1_l = None
        self.sample_v1_r = None

        # === Test Actions ===

        self.compute_p_h0_l = None
        self.compute_p_h0_r = None

        self.sample_h0_l = None
        self.sample_h0_r = None

        self.sample_v1_l = None
        self.sample_v1_r = None
        self.sample_v = None

        self.compute_visible_from_hidden_l = None
        self.compute_visible_from_hidden_r = None
        self.compute_visible_from_hidden = None

        # === Init Vars and Actions ===

        self._initialize_vars()

        # === Check After Init ===

        assert self.compute_p_h0_l is not None
        assert self.compute_p_h0_r is not None

        assert self.sample_h0_l is not None
        assert self.sample_h0_r is not None

        assert self.sample_v1_l is not None
        assert self.sample_v1_r is not None
        assert self.sample_v is not None

        assert self.compute_visible_from_hidden_l is not None
        assert self.compute_visible_from_hidden_r is not None
        assert self.compute_visible_from_hidden is not None

        assert self.update_weights is not None
        assert self.update_deltas is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm_m = tf.nn.l2_normalize(self.sample_v, 1)
            cos_val_m = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm_m), 1))

            self.compute_err_m = tf.acos(cos_val_m) / tf.constant(np.pi)
        else:
            self.compute_err_m = tf.reduce_mean(tf.square((self.x - self.sample_v)))

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err_m, feed_dict={self.x: batch_x})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return (self.sess.run(self.compute_p_h0_l, feed_dict={self.x: batch_x}),
                self.sess.run(self.sample_h0_r, feed_dict={self.x: batch_x}))

    def transform_inv(self, batch_y):
        return (self.sess.run(self.compute_visible_from_hidden_l, feed_dict={self.y: batch_y}),
                self.sess.run(self.compute_visible_from_hidden_r, feed_dict={self.y: batch_y}))

    def reconstruct(self, batch_x):
        return self.sess.run(self.sample_v, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

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

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        errs = []

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
                self.partial_fit(batch_x)

                batch_err = self.get_err(batch_x)

                epoch_errs[epoch_errs_ptr] = batch_err

                epoch_errs_ptr += 1

            if verbose:

                err_mean = epoch_errs.mean()

                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

    def get_weights(self):
        return (
            self.sess.run(self.w),
            self.sess.run(self.w_l),
            self.sess.run(self.w_r),
            self.sess.run(self.bv),
            self.sess.run(self.bv_l),
            self.sess.run(self.bv_r),
            self.sess.run(self.bh),
            self.sess.run(self.bh_l),
            self.sess.run(self.bh_r)
        )

    def save_weights(self, filename, name):
        # saver = tf.train.Saver({
        #     name + '_w_l': self.w_l,
        #     name + '_w_r': self.w_r,
        #     name + '_v_l': self.visible_bias_l,
        #     name + '_v_r': self.visible_bias_r,
        #     name + '_h_l': self.hidden_bias_l,
        #     name + '_h_r': self.hidden_bias_r,
        # })
        # return saver.save(self.sess, filename)

        # data = {
        #     name + '_w_l': self.w_l.eval(session=self.sess).tolist(),
        #     name + '_w_r': self.w_r.eval(session=self.sess).tolist(),
        #     name + '_v_l': self.visible_bias_l.eval(session=self.sess).tolist(),
        #     name + '_v_r': self.visible_bias_r.eval(session=self.sess).tolist(),
        #     name + '_h_l': self.hidden_bias_l.eval(session=self.sess).tolist(),
        #     name + '_h_r': self.hidden_bias_r.eval(session=self.sess).tolist(),
        # }
        data = {
            name + '_w_l': self.w_l.eval(session=self.sess).tolist(),
            name + '_w_r': self.w_r.eval(session=self.sess).tolist(),
            name + '_v_l': self.bv_l.eval(session=self.sess).tolist(),
            name + '_v_r': self.bv_r.eval(session=self.sess).tolist(),
            name + '_h_l': self.bh_l.eval(session=self.sess).tolist(),
            name + '_h_r': self.bh_r.eval(session=self.sess).tolist(),
        }

        import json

        d = json.dumps(data)

        with open(filename, "w") as f:
            f.write(d)

    def set_weights(self, w_l, w_r, visible_bias_l, visible_bias_r, hidden_bias_l, hidden_bias_r):
        self.sess.run(self.w_l.assign(w_l))
        self.sess.run(self.w_r.assign(w_r))
        self.sess.run(self.bv_l.assign(visible_bias_l))
        self.sess.run(self.bv_r.assign(visible_bias_r))
        self.sess.run(self.bh_l.assign(hidden_bias_l))
        self.sess.run(self.bh_r.assign(hidden_bias_r))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({
            name + '_w_l': self.w_l,
            name + '_w_r': self.w_r,
            name + '_v_l': self.bv_l,
            name + '_v_r': self.bv_r,
            name + '_h_l': self.bh_l,
            name + '_h_r': self.bh_r,
        })
        saver.restore(self.sess, filename)

    def _initialize_vars(self):

        def f(old_delta, new_delta):
            # return (self.momentum * old_delta +
            #         self.learning_rate * new_delta * (1 - self.momentum) / tf.to_float(tf.shape(new_delta)[0]))
            return self.learning_rate * new_delta

        p_h0_l = tf.nn.sigmoid(tf.matmul(self.x, self.w_l) + self.bh_l)
        p_h0_r = tf.nn.sigmoid(tf.matmul(self.x, self.w_r) + self.bh_r)

        s_h0_l = sample_bernoulli(p_h0_l)
        s_h0_r = sample_bernoulli(p_h0_r)

        p_v1_l = tf.nn.sigmoid(tf.matmul(s_h0_l, tf.transpose(self.w_l)) + self.bv_l)
        p_v1_r = tf.nn.sigmoid(tf.matmul(s_h0_r, tf.transpose(self.w_r)) + self.bv_r)

        s_v1_l = sample_bernoulli(p_v1_l)
        s_v1_r = sample_bernoulli(p_v1_r)

        p_h1_l = tf.nn.sigmoid(tf.matmul(s_v1_l, self.w_l) + self.bh_l)
        p_h1_r = tf.nn.sigmoid(tf.matmul(s_v1_r, self.w_r) + self.bh_r)

        positive_grad_l = tf.matmul(tf.transpose(self.x), p_h0_l)
        positive_grad_r = tf.matmul(tf.transpose(self.x), p_h0_r)

        negative_grad_l = tf.matmul(tf.transpose(s_v1_l), p_h1_l)
        negative_grad_r = tf.matmul(tf.transpose(s_v1_r), p_h1_r)

        delta_w_new_l = f(self.delta_w_l, positive_grad_l - negative_grad_l)
        delta_w_new_r = f(self.delta_w_r, positive_grad_r - negative_grad_r)

        delta_visible_bias_new_l = f(self.delta_bv_l, tf.reduce_mean(self.x - p_v1_l, 0))
        delta_visible_bias_new_r = f(self.delta_bv_r, tf.reduce_mean(self.x - p_v1_r, 0))

        delta_hidden_bias_new_l = f(self.delta_bh_l, tf.reduce_mean(p_h0_l - p_h1_l, 0))
        delta_hidden_bias_new_r = f(self.delta_bh_r, tf.reduce_mean(p_h0_r - p_h1_r, 0))

        update_delta_w_l = self.delta_w_l.assign(delta_w_new_l)
        update_delta_w_r = self.delta_w_r.assign(delta_w_new_r)

        update_delta_bv_l = self.delta_bv_l.assign(delta_visible_bias_new_l)
        update_delta_bv_r = self.delta_bv_r.assign(delta_visible_bias_new_r)

        update_delta_bh_l = self.delta_bh_l.assign(delta_hidden_bias_new_l)
        update_delta_bh_r = self.delta_bh_r.assign(delta_hidden_bias_new_r)

        update_w_l = self.w_l.assign(self.w_l + delta_w_new_l)
        update_w_r = self.w_r.assign(self.w_r + delta_w_new_r)

        update_bv_l = self.bv_l.assign(self.bv_l + delta_visible_bias_new_l)
        update_bv_r = self.bv_r.assign(self.bv_r + delta_visible_bias_new_r)

        update_bh_l = self.bh_l.assign(self.bh_l + delta_hidden_bias_new_l)
        update_bh_r = self.bh_r.assign(self.bh_r + delta_hidden_bias_new_r)

        # Actions

        self.update_deltas = [
            update_delta_w_l, update_delta_bv_l, update_delta_bh_l,
            update_delta_w_r, update_delta_bv_r, update_delta_bh_r
        ]

        self.update_weights = [
            update_w_l, update_bv_l, update_bh_l,
            update_w_r, update_bv_r, update_bh_r
        ]

        self.compute_p_h0_l = tf.nn.sigmoid(tf.matmul(self.x, self.w_l) + self.bh_l)
        self.compute_p_h0_r = tf.nn.sigmoid(tf.matmul(self.x, self.w_r) + self.bh_r)

        self.sample_h0_l = sample_bernoulli(self.compute_p_h0_l)
        self.sample_h0_r = sample_bernoulli(self.compute_p_h0_r)

        self.compute_p_v1_l = tf.nn.sigmoid(tf.matmul(self.compute_p_h0_l, tf.transpose(self.w_l)) + self.bv_l)
        self.compute_p_v1_r = tf.nn.sigmoid(tf.matmul(self.compute_p_h0_r, tf.transpose(self.w_r)) + self.bv_r)

        self.sample_v1_l = sample_bernoulli(self.compute_p_v1_l)
        self.sample_v1_r = sample_bernoulli(self.compute_p_v1_r)

        # self.sample_v = tf.add(self.compute_p_v1_l, self.compute_p_v1_r) / 2
        self.sample_v = tf.divide(
            tf.add(self.sample_v1_l, self.sample_v1_r), 2)

        self.compute_visible_from_hidden_l = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w_l)) + self.bv_l)
        self.compute_visible_from_hidden_r = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w_r)) + self.bv_r)
        self.compute_visible_from_hidden = tf.divide(
            tf.add(self.compute_visible_from_hidden_l, self.compute_visible_from_hidden_r), 2)
