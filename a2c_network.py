# a2c_network.py loads up the trained A2C model into its own tf session, and allows external modules
# to call its inference function to observe the network outputs, this approach was used as it makes
# using multiple neural networks in the same job simpler
# however the approach of handling rnn_state outside of a2c_network currently is not ideal
import tensorflow as tf
import os
import numpy as np

N_S = 4
N_A = 1
LSTM_UNITS = 16
HN_A = 50
HN_C = 200
TRAJECTORY_LENGTH = 80
A_BOUND = [-1, 1]
MODEL_FILE = 'model-ep-2500-finalr-18541.ckpt'
DATA_DIR = './data/'
LOG_DIR = '.models//rl_models/'

class ACNetwork(object):
    """implements the AC network model for estimating vehicle host actions"""

    def __init__(self):
        # set up tf session and model
        rl_graph = tf.Graph()
        rl_config = tf.ConfigProto()
        rl_config.gpu_options.allow_growth = True
        self.sess_rl = tf.Session(graph=rl_graph, config=rl_config)

        with self.sess_rl.as_default():
            with rl_graph.as_default():
                # BUILD MODEL
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.mu, self.sigma, self.v, self.a_params, self.c_params = \
                    self._build_net()  # parameters of AC net
                # Scale mu to action space, and add small value to sigma to avoid NaN errors
                with tf.name_scope('wrap_a_out'):
                    self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-4

                # Normal distribution with location = mu, scale = sigma
                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

                # Choose action
                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0],
                                              A_BOUND[1])  # sample a action from distribution

                # GET SAVED WEIGHTS
                saver = tf.train.Saver()
                checkpoint_path = os.path.join(LOG_DIR, MODEL_FILE)
                saver.restore(self.sess_rl, checkpoint_path)
        print('rl_model: Restored model: %s' % MODEL_FILE)

    # Build the network
    def _build_net(self):  # neural network structure of the actor and critic
        w_init = tf.random_normal_initializer(0., .1)
        # Actor network
        with tf.variable_scope('Global_Net/actor'):
            # hidden layer
            l1_a = tf.layers.dense(self.s, HN_A, tf.nn.relu6, kernel_initializer=w_init, name='l1a')
            l2_a = tf.layers.dense(l1_a, HN_A, tf.nn.relu6, kernel_initializer=w_init, name='l2a')
            l3_a = tf.layers.dense(l2_a, HN_A, tf.nn.relu6, kernel_initializer=w_init, name='l3a')

            # Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(l3_a, [0])
            step_size = tf.shape(self.s)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, LSTM_UNITS])

            # expected action value
            mu = tf.layers.dense(rnn_out, N_A, tf.nn.tanh, kernel_initializer=w_init,
                                 name='mu')  # estimated action value
            # expected variance
            sigma = tf.layers.dense(rnn_out, N_A, tf.nn.softplus, kernel_initializer=w_init,
                                    name='sigma')  # estimated variance

        # Critic network
        with tf.variable_scope('Global_Net/critic'):
            l_c = tf.layers.dense(self.s, HN_C, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # estimated value for state

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Global_Net/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Global_Net/critic')
        return mu, sigma, v, a_params, c_params

    def inference(self, s, rnn_state):
        s = np.reshape(s, (1, N_S))  # reshape state vector
        rnn_state = self.sess_rl.run(self.state_out, {self.s: s,
                                                 self.state_in[0]: rnn_state[0],
                                                 self.state_in[1]: rnn_state[1]})       # update rnn state
        return rnn_state, self.sess_rl.run(self.A, {self.s: s, self.state_in[0]: rnn_state[0],     # choose action
                                          self.state_in[1]: rnn_state[1]})[0]
