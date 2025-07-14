import math
import numpy as np
import tensorflow.compat.v1 as tf
import tflearn

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
# PPO2
EPS = 0.2


class Network:
    def CreateNetwork(self, inputs):
        with tf.variable_scope("actor"):
            lstm_out = tflearn.lstm(inputs, FEATURE_NUM, dropout=0.5)
            fc1 = tflearn.fully_connected(lstm_out, FEATURE_NUM * 2, activation="relu")
            fc1 = tflearn.dropout(fc1, 0.5)
            fc2 = tflearn.fully_connected(fc1, FEATURE_NUM, activation="relu")
            pi = tflearn.fully_connected(fc2, self.a_dim, activation="softmax")

        with tf.variable_scope("critic"):
            lstm_out_v = tflearn.lstm(inputs, FEATURE_NUM, dropout=0.5)
            fc1_v = tflearn.fully_connected(lstm_out_v, FEATURE_NUM * 2, activation="relu")
            fc1_v = tflearn.dropout(fc1_v, 0.5)
            fc2_v = tflearn.fully_connected(fc1_v, FEATURE_NUM, activation="relu")
            value = tflearn.fully_connected(fc2_v, 1, activation="linear")

        return pi, value



    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(
            self.set_network_params_op,
            feed_dict={
                i: d for i, d in zip(self.input_network_params, input_network_params)
            },
        )

    def r(self, pi_new, pi_old, acts):
        return tf.reduce_sum(
            tf.multiply(pi_new, acts), reduction_indices=1, keepdims=True
        ) / tf.reduce_sum(tf.multiply(pi_old, acts), reduction_indices=1, keepdims=True)

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self._entropy_weight = np.log(self.a_dim)
        self.H_target = 0.1

        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.entropy_weight = tf.placeholder(tf.float32)
        self.pi, self.val = self.CreateNetwork(inputs=self.inputs)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1.0 - ACTION_EPS)

        self.entropy = -tf.reduce_sum(
            tf.multiply(self.real_out, tf.log(self.real_out)),
            reduction_indices=1,
            keepdims=True,
        )
        self.adv = tf.stop_gradient(self.R - self.val)
        self.ppo2loss = tf.minimum(
            self.r(self.real_out, self.old_pi, self.acts) * self.adv,
            tf.clip_by_value(
                self.r(self.real_out, self.old_pi, self.acts), 1 - EPS, 1 + EPS
            )
            * self.adv,
        )
        self.dual_loss = tf.where(
            tf.less(self.adv, 0.0),
            tf.maximum(self.ppo2loss, 3.0 * self.adv),
            self.ppo2loss,
        )

        # Get all network parameters
        self.network_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor"
        )
        self.network_params += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic"
        )

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape())
            )
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        self.policy_loss = -tf.reduce_sum(
            self.dual_loss
        ) - self.entropy_weight * tf.reduce_sum(self.entropy)
        self.policy_opt = tf.train.AdamOptimizer(self.lr_rate).minimize(
            self.policy_loss
        )
        self.val_loss = tflearn.mean_square(self.val, self.R)
        self.val_opt = tf.train.AdamOptimizer(self.lr_rate * 10.0).minimize(
            self.val_loss
        )

    def predict(self, input):
        assert input.shape[1:] == (self.s_dim[1], self.s_dim[0]), f"Esperado shape (S_LEN, S_INFO), recebeu {input.shape[1:]}"

        action = self.sess.run(self.real_out, feed_dict={self.inputs: input})
        return action[0]

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):

        self.sess.run(
            [self.policy_opt, self.val_opt],
            feed_dict={
                self.inputs: s_batch,
                self.acts: a_batch,
                self.R: v_batch,
                self.old_pi: p_batch,
                self.entropy_weight: self._entropy_weight,
            },
        )
        # adaptive entropy weight
        # https://arxiv.org/abs/2003.13590
        p_batch = np.clip(p_batch, ACTION_EPS, 1.0 - ACTION_EPS)
        _H = np.mean(np.sum(-np.log(p_batch) * p_batch, axis=1))
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:
            v_batch = self.sess.run(self.val, feed_dict={self.inputs: s_batch})
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")