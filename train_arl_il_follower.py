# Author: Sampo Kuutti
# Organisation: University of Surrey / Connected & Autonomous Vehicles Lab
# Email: s.j.kuutti@surrey.ac.uk
# train_arl_* trains an adversarial agent based on A2C Reinforcement Learning as the lead vehicle to a
# learning based follower vehicle, and attempts to create collisions with it
import numpy as np
import os
import tensorflow as tf
import csv
import random
import ipg_proxy
from collections import deque
import time
import datetime
import argparse
import sl_network
import math

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H_%M_%S')
FPATH = '/vol/research/safeav/Sampo/condor-a2c/test/log_arl/'  # use project directory path
# PARAMETERS
OUTPUT_GRAPH = True  # graph output
RENDER = True  # render one worker
RENDER_EVERY = 100   # render every N episodes
LOG_DIR = timestamp  # save location for logs
N_WORKERS = 1  # number of workers
MAX_EP_STEP = 200  # maximum number of steps per episode (unless another limit is used)
MAX_GLOBAL_EP = 2500  # total number of episodes
MAX_PROXY_EP = 2500      # total number of episodes to train on proxy, before switching to ipg simulations
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 80  # sets how often the global net is updated
GAMMA = 0.99  # discount factor
ENTROPY_BETA = 1e-3  # entropy factor
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
SAFETY_ON = 0   # safety cages, 0 = disabled 1 = enabled
REPLAY_MEMORY_CAPACITY = int(1e4)  # capacity of experience replay memory
TRAUMA_MEMORY_CAPACITY = int(1e2)  # capacity of trauma memory
MINIBATCH_SIZE = 64  # size of the minibatch for training with experience replay
TRAJECTORY_LENGTH = 80  # size of the trajectory used in weight updates
UPDATE_ENDSTEP = True  # update at the end of episode using previous MB_SIZE experiences
UPDATE_TRAUMA = 16       # update weights using the trauma memory every UPDATE_TRAUMA updates
OFF_POLICY = True       # update off-policy using ER/TM
ON_POLICY = True        # update on-policy using online experiences
CHECKPOINT_EVERY = 100  # sets how often to save weights
HN_A = 50   # hidden neurons for actor network
HN_C = 200  # hidden neurons for critic network
LSTM_UNITS = 16     # lstm units in actor network
MAX_GRAD_NORM = 0.5     # max l2 grad norm for gradient clipping
V_MIN = 17      # minimum lead vehicle velocity (m/s)
V_MAX = 30      # maximum lead vehicle velocity (m/s)
# Action Space Shape
N_S = 4  # number of states
N_A = 1  # number of actions
A_BOUND = [-6, 2]  # action bounds


def get_arguments():
    parser = argparse.ArgumentParser(description='RL training')
    parser.add_argument(
        '--lr_a',
        type=float,
        default=LR_A,
        help='Actor learning rate'
    )
    parser.add_argument(
        '--lr_c',
        type=float,
        default=LR_C,
        help='Critic learning rate'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=GAMMA,
        help='Discount rate gamma'
    )
    parser.add_argument(
        '--max_eps',
        type=int,
        default=MAX_GLOBAL_EP,
        help='Checkpoint file to restore model weights from.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=MINIBATCH_SIZE,
        help='Batch size. Must divide evenly into dataset sizes.'
    )
    parser.add_argument(
        '--trajectory',
        type=float,
        default=TRAJECTORY_LENGTH,
        help='Length of trajectories in minibatches'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=CHECKPOINT_EVERY,
        help='Number of steps before checkpoint.'
    )
    parser.add_argument(
        '--ent_beta',
        type=float,
        default=ENTROPY_BETA,
        help='Entropy coefficient beta'
    )
    parser.add_argument(
        '--fpath',
        type=str,
        default=FPATH,
        help='File path to root folder.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=LOG_DIR,
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--store_metadata',
        type=bool,
        default=False,
        help='Storing debug information for TensorBoard.'
    )
    parser.add_argument(
        '--restore_from',
        type=str,
        default=None,
        help='Checkpoint file to restore model weights from.'
    )
    parser.add_argument(
        '--hn_a',
        type=int,
        default=HN_A,
        help='Number of hidden neurons in actor network.'
    )
    parser.add_argument(
        '--hn_c',
        type=int,
        default=HN_C,
        help='Number of hidden neurons in critic network.'
    )
    parser.add_argument(
        '--lstm_units',
        type=int,
        default=LSTM_UNITS,
        help='Number of lstm cells in actor network.'
    )
    parser.add_argument(
        '--store_results',
        action='store_true',
        help='Storing episode results in csv files.'
    )
    parser.add_argument(
        '--trauma',
        action='store_true',
        help='If true use trauma memory in off-policy updates.'
    )
    parser.add_argument(
        '--max_norm',
        type=float,
        default=MAX_GRAD_NORM,
        help='Maximum L2 norm of the gradient for gradient clipping.'
    )
    parser.add_argument(
        '--v_min',
        type=int,
        default=V_MIN,
        help='Minimum lead vehicle velocity (m/s).'
    )
    parser.add_argument(
        '--v_max',
        type=int,
        default=V_MAX,
        help='Maximum lead vehicle velocity (m/s).'
    )

    return parser.parse_args()


# reward function based on inverse time headway and large pay off for crashes
def calculate_reward3(t_h):

    if t_h > 0:  # positive time headway
        r = 1 / t_h
    else:  # crash occurred
        r = 100

    # cap r at 100 (for t_h < 0.01s)
    if r > 100:
        r = 100

    return r


# replay memory
replay_memory = deque(maxlen=REPLAY_MEMORY_CAPACITY)  # used for O(1) popleft() operation


def add_to_memory(experience):
    replay_memory.append(experience)


def sample_from_memory(minibatch_size):
    return random.sample(replay_memory, minibatch_size)


# trauma memory
trauma_buffer = deque(maxlen=TRAJECTORY_LENGTH)
trauma_memory = deque(maxlen=TRAUMA_MEMORY_CAPACITY)


def add_to_trauma(experience):
    trauma_memory.append(experience)


def sample_from_trauma(minibatch_size):
    return random.sample(trauma_memory, minibatch_size)


# Network for the Actor Critic
class ACNet(object):
    def __init__(self, args, scope, sess, globalAC=None):
        self.sess = sess
        self.actor_optimizer = tf.train.RMSPropOptimizer(args.lr_a, name='RMSProp_Actor')  # optimizer for the actor
        self.critic_optimizer = tf.train.RMSPropOptimizer(args.lr_c, name='RMSProp_Critic')  # optimizer for the critic

        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_params, self.c_params = self._build_net(args, scope)[-2:]  # parameters of actor and critic net
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')  # action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # v_target value

                self.mu, self.sigma, self.v, self.a_params, self.c_params = self._build_net(args,
                    scope)  # get mu and sigma of estimated action from neural net

                # advantage function A(s) = V_target(s) - V(s)
                td = tf.subtract(self.v_target, self.v, name='TD_error')

                # Critic Loss
                with tf.name_scope('c_loss'):
                    # value loss L_c = (R - V(s))^2
                    self.c_loss = tf.reduce_mean(tf.square(td))

                # Scale mu to action space, and add small value to sigma to avoid NaN errors
                with tf.name_scope('wrap_action'):
                    # use abs value of A_BOUND[0] as it is bigger than A_BOUND[1]
                    # The action value is later clipped so values outside of A_BOUND[1] will be constrained
                    self.mu, self.sigma = self.mu * (-A_BOUND[0]), self.sigma + 1e-4

                # Normal distribution with location = mu, scale = sigma
                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

                # Actor loss
                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    # Entropy H(s) = 0.5(log(2*pi*sigma^2)+1) see: https://arxiv.org/pdf/1602.01783.pdf page 13
                    entropy = normal_dist.entropy()  # encourage exploration
                    # policy loss L_a = A(s,a) * -logpi(a|s) - B * H(s)
                    self.a_loss = tf.reduce_mean(-(args.ent_beta * entropy + log_prob * td))

                # Choose action
                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0],
                                              A_BOUND[1])  # sample a action from distribution

                # Compute the gradients
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss,
                                                self.a_params)  # calculate gradients for the network weights
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    # clip gradients by global norm
                    self.a_grads, a_grad_norm = tf.clip_by_global_norm(self.a_grads, MAX_GRAD_NORM)
                    self.c_grads, c_grad_norm = tf.clip_by_global_norm(self.c_grads, MAX_GRAD_NORM)

            # Update weights
            with tf.name_scope('sync'):  # update local and global network weights
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

    # Build the network
    def _build_net(self, args, scope):  # neural network structure of the actor and critic
        w_init = tf.random_normal_initializer(0., .1)
        # Actor network
        with tf.variable_scope('actor'):
            # hidden layer
            l1_a = tf.layers.dense(self.s, args.hn_a, tf.nn.relu6, kernel_initializer=w_init, name='l1a')
            l2_a = tf.layers.dense(l1_a, args.hn_a, tf.nn.relu6, kernel_initializer=w_init, name='l2a')
            l3_a = tf.layers.dense(l2_a, args.hn_a, tf.nn.relu6, kernel_initializer=w_init, name='l3a')

            # Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.LSTMCell(args.lstm_units, state_is_tuple=True)
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
            rnn_out = tf.reshape(lstm_outputs, [-1, args.lstm_units])

            # expected action value
            mu = tf.layers.dense(rnn_out, N_A, tf.nn.tanh, kernel_initializer=w_init,
                                 name='mu')  # estimated action value
            # expected variance
            sigma = tf.layers.dense(rnn_out, N_A, tf.nn.softplus, kernel_initializer=w_init,
                                    name='sigma')  # estimated variance

        # Critic network
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, args.hn_c, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # estimated value for state

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, rnn_state):  # run by a local
        s = np.reshape(s, (1, N_S))    # reshape state vector
        return self.sess.run(self.A, {self.s: s, self.state_in[0]: rnn_state[0],
                                      self.state_in[1]: rnn_state[1]})[0]



class ProxyEnv(object):
    def __init__(self, args):
        # Define proxy environment (See ITSC'19 for details)
        self.proxy = ipg_proxy.IpgProxy()

        self.args = args  # get arguments

        # set states to zero
        self.v_rel = 0.0
        self.v = 0.0
        self.x_rel = 0.0
        self.a = 0.0
        self.t = 0.0
        self.t_h = 0.0
        self.x = 0.0
        self.x_lead = 0.0
        self.v_lead = 0.0
        self.a_lead = 0.0

        self.ep = 0  # episode
        self.arr_scen = []  # scenarios used
        self.randomise_cof = False  # coefficient of friction randomisation
        self.prev_action = 0.0  # previous action of host vehicle, used for the proxy network

    def reset(self):
        # set initial states
        self.t = 0.0
        self.v = 25.5  # 91.8 km/h
        self.a = 0.0
        self.x = 5.0
        self.prev_action = 0.0

        # lead vehicle states
        self.x_lead = 55.0  # longitudinal position
        self.v_lead = float(random.randint(25, 32))  # velocity, randomly chosen between 25 and 32 m/s
        self.a_lead = 0.0  # acceleration
        self.v_rel = self.v_lead - self.v  # relative velocity
        self.x_rel = self.x_lead - self.x  # relative distance
        if self.v != 0:  # check for division by 0
            self.t_h = self.x_rel / self.v
        else:
            self.t_h = self.x_rel

        # empty arrays
        self.arr_a = []  # acceleration array
        self.arr_j = []  # jerk array
        self.arr_t = []  # time array
        self.arr_x = []  # x_rel array
        self.arr_v = []  # velocity array
        self.arr_dv = []  # relative velocity array
        self.arr_th = []  # time headway array
        self.arr_y_0 = []  # original output
        self.arr_y_sc = []  # safety cage output (arr_y_sc and arr_sc are not particularly used here,
        # but we populate them to keep logs in same format as previous experiments with SC used
        self.arr_sc = []  # safety cage number
        self.arr_cof = []  # coefficient of friction
        self.arr_v_leader = []  # lead vehicle velocity
        self.arr_a_leader = []  # lead vehicle acceleration
        self.arr_rewards = []  # rewards
        # lead vehicle states
        T_lead = []
        X_lead = []
        V_lead = []
        A_lead = []

        # load test run
        if self.randomise_cof:
            # Option 1: Use random coefficients of frictions
            scen = random.randint(1, 25)
        else:
            # Option 2: Use a pre-determined list of coefficient of frictions
            with open('./traffic_data/' + 'scens.csv') as f:
               reader = csv.DictReader(f, delimiter=',')
               for row in reader:
                   self.arr_scen.append(float(row['s']))  # test run id
        scen = int(self.arr_scen[self.ep - 1])
        self.cof = 0.375 + scen * 0.025  # calculate coefficient of friction

        # define states
        states = [self.v_rel, self.t_h, self.v, self.a]
        states_follower = [self.v_rel, self.t_h, self.v]

        return states, states_follower

    def step(self, action_arl, action_follower):
        self.a_lead = float(action_arl)
        self.v_lead = self.v_lead + (self.a_lead * 0.04)
        self.x_lead = self.x_lead + (self.v_lead * 0.04)
        # constraints
        if self.v_lead > self.args.v_max:
            v_lead = float(self.args.v_max)
        elif self.v_lead < self.args.v_min:
            self.v_lead = float(self.args.v_min)

        self.t = self.t + 0.04  # time

        # use proxy for host vehicle states from gas and brake pedals
        proxy_out = self.proxy.inference([self.v, self.a, self.cof, action_follower, self.prev_action])  # proxy_out infers the v_t+1

        v_ = float(proxy_out)  # host velocity
        delta_v = v_ - self.v  # calculate delta_v
        if delta_v > 0.4:  # limit a to +/- 10m/s^2
            delta_v = 0.4
            v_ = delta_v + self.v
        elif delta_v < -0.4:
            delta_v = -0.4
            v_ = delta_v + self.v
        if v_ < 0:  # check for negative velocity
            v_ = 0
        self.v = v_
        self.a = delta_v / 0.04  # host longitudinal acceleration
        self.x = self.x + (self.v * 0.04)  # host longitudinal position
        # print('t = %f, y = %f, v = %f, a = %f, x = %f' % (t, output, v, a, x))

        # relative states
        self.v_rel = float(self.v_lead - self.v)  # relative velocity
        self.x_rel = float(self.x_lead - self.x)  # relative distance

        # enter variables into arrays
        self.arr_a.append(self.a)
        self.arr_t.append(self.t)
        self.arr_x.append(self.x_rel)
        self.arr_v.append(self.v)
        self.arr_dv.append(self.v_rel)
        self.arr_th.append(self.t_h)
        self.arr_cof.append(self.cof)

        self.arr_v_leader.append(self.v_lead)
        self.arr_a_leader.append(self.a_lead)

        # calculate time headway
        if self.v != 0:
            self.t_h = self.x_rel / self.v
        else:
            self.t_h = self.x_rel

        # calculate reward
        reward = calculate_reward3(self.t_h)

        self.prev_action = action_follower

        # populate arrays for logs
        self.arr_y_0.append(float(action_follower))
        self.arr_y_sc.append(float(action_follower))
        self.arr_sc.append(0)
        self.arr_rewards.append(reward)

        # check for crash
        if self.x_rel <= 0.0:
            crash = 1
        else:
            crash = 0

        # check for terminal state
        if self.t >= 300 or crash:
            done = True
        else:
            done = False

        # define states
        states = [self.v_rel, self.t_h, self.v, self.a]
        states_follower = [self.v_rel, self.t_h, self.v]

        return states, states_follower, reward, done, crash


# A3C Worker agent, creates its own copy of the actor-critic network and environment
class Worker(object):
    def __init__(self, args, name, globalAC, sess):
        self.name = name
        self.AC = ACNet(args, name, sess, globalAC)  # create ACNet for each worker
        self.sess = sess

    def work(self):
        global global_rewards, global_episodes
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        # Initialise environment
        env = ProxyEnv(args)

        # Define target model for testing
        sl_net = sl_network.SupervisedNetwork()

        crash_count = 0  # count number of crashes in training run

        # loop episodes
        while global_episodes < args.max_eps:
            # initialise rnn state
            rnn_state = self.AC.state_init
            self.batch_rnn_state = rnn_state

            ER_buffer = []  # experience replay buffer
            trauma_buffer.clear()  # clear trauma buffer

            print('\ntest no. %d' % global_episodes)

            # Run training using Ipg Proxy
            ep_r = 0  # set ep reward to 0

            states, states_host = env.reset()
            done = False
            crash = 0

            # loop time-steps
            while not done:  # check if simulation is running
                # evaluate neural network output
                # rnn state
                rnn_state = sess.run(self.AC.state_out, {self.AC.s: np.reshape(states, (1, N_S)),
                                                         self.AC.state_in[0]: rnn_state[0],
                                                         self.AC.state_in[1]: rnn_state[1]})
                # action for adversarial agent
                action_arl = self.AC.choose_action(states, rnn_state)

                # action for host vehicle action
                action_host = sl_net.inference(states_host)

                # get new states and increment environment step
                new_states, new_states_host, reward, done, crash = env.step(action_arl, action_host)

                # calculate reward
                ep_r += reward

                # add to trauma memory buffer
                trauma_buffer.append((states, action_arl, reward, new_states))

                # stop simulation if a crash occurs
                if crash:
                    crash_count += 1
                    print('crash occurred: simulation run stopped')
                    if len(trauma_buffer) >= TRAJECTORY_LENGTH:
                        add_to_trauma(trauma_buffer)

                # update buffers
                buffer_s.append(states)
                buffer_a.append(action_arl)
                buffer_r.append(reward)

                ER_buffer.append((states, action_arl, reward, new_states))
                # if buffer > mb_size add to experience replay and empty buffer
                if len(ER_buffer) >= args.trajectory:
                    add_to_memory(ER_buffer)
                    ER_buffer = []

                # update weights
                if total_step % UPDATE_GLOBAL_ITER == 0:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal state
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: np.reshape(new_states, (1, N_S))})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + args.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.state_in[0]: self.batch_rnn_state[0],
                        self.AC.state_in[1]: self.batch_rnn_state[1]
                    }
                    self.batch_rnn_state = sess.run(self.AC.state_out,
                                                    feed_dict=feed_dict)  # update rnn state, run training step
                    self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()  # get global parameters to local ACNet

                # update state variables
                states = new_states

                total_step += 1

            # Run an update step at the end of episode
            if UPDATE_ENDSTEP:
                minibatch = trauma_buffer
                batch_s = np.asarray([elem[0] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_S)
                batch_a = np.asarray([elem[1] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_A)
                batch_r = np.asarray([elem[2] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, 1)

                # Generalised Advantage Estimation GAE:
                v_s_ = 0  # terminal state
                batch_v_target = []
                for r in batch_r[::-1]:  # reverse buffer r
                    v_s_ = r + args.gamma * v_s_
                    batch_v_target.append(v_s_)
                batch_v_target.reverse()

                feed_dict = {
                    self.AC.s: batch_s,
                    self.AC.a_his: batch_a,
                    self.AC.v_target: batch_v_target,
                    # self.AC.next_s: np.asarray([elem[3] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_S),
                    self.AC.state_in[0]: self.batch_rnn_state[0],
                    self.AC.state_in[1]: self.batch_rnn_state[1]
                }

                self.batch_rnn_state = sess.run(self.AC.state_out,
                                                feed_dict=feed_dict)  # update rnn state
                self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                self.AC.pull_global()  # get global parameters to local ACNet

            if OFF_POLICY:
                for off_pol_i in range(0, args.batch_size):
                    if args.trauma and off_pol_i == 0 and len(trauma_memory) >= 1:  # run one update from trauma memory
                        minibatch = sample_from_trauma(1)[-1]
                    else:
                        # grab N (s,a,r,s') tuples from replay memory
                        minibatch = sample_from_memory(1)[-1]  # sample and flatten minibatch

                    # reset lstm cell state
                    rnn_state = self.AC.state_init
                    self.batch_rnn_state = rnn_state

                    batch_s = np.asarray([elem[0] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_S)
                    batch_a = np.asarray([elem[1] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_A)
                    batch_r = np.asarray([elem[2] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, 1)

                    # Generalised Advantage Estimation GAE:
                    v_s_ = self.sess.run(self.AC.v, {self.AC.s: np.reshape(batch_s[-1], (1, N_S))})[0, 0]
                    batch_v_target = []
                    for r in batch_r[::-1]:  # reverse buffer r
                        v_s_ = r + args.gamma * v_s_
                        batch_v_target.append(v_s_)
                    batch_v_target.reverse()

                    # create feed dict
                    feed_dict = {
                        self.AC.s: batch_s,
                        self.AC.a_his: batch_a,
                        self.AC.v_target: batch_v_target,
                        # self.AC.next_s: np.asarray([elem[3] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, 3),
                        self.AC.state_in[0]: self.batch_rnn_state[0],
                        self.AC.state_in[1]: self.batch_rnn_state[1]
                    }

                    # update parameters
                    self.batch_rnn_state = sess.run(self.AC.state_out,
                                                    feed_dict=feed_dict)  # update rnn state
                    self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    self.AC.pull_global()  # get global parameters to local ACNet

                    # reset lstm cell state
                    rnn_state = self.AC.state_init
                    self.batch_rnn_state = rnn_state

            buffer_s, buffer_a, buffer_r = [], [], []  # empty buffers

            # Update summaries and print episode performance before starting next episode
            # update tensorboard summaries
            summary = sess.run(merged, feed_dict=feed_dict)
            writer.add_summary(summary, global_episodes)
            writer.flush()
            perf_summary = tf.Summary(value=[tf.Summary.Value(tag='Perf/Reward', simple_value=float(ep_r))])
            writer.add_summary(perf_summary, global_episodes)
            writer.flush()
            perf_summary = tf.Summary(value=[tf.Summary.Value(tag='Perf/Mean_Reward', simple_value=float(np.mean(env.arr_rewards)))])
            writer.add_summary(perf_summary, global_episodes)
            writer.flush()
            perf_summary = tf.Summary(value=[tf.Summary.Value(tag='Perf/Mean_Th', simple_value=float(np.mean(env.arr_th)))])
            writer.add_summary(perf_summary, global_episodes)
            writer.flush()
            perf_summary = tf.Summary(value=[tf.Summary.Value(tag='Perf/Min_Th', simple_value=float(np.min(env.arr_th)))])
            writer.add_summary(perf_summary, global_episodes)
            writer.flush()

            # append episode reward to list
            global_rewards.append(ep_r)

            # print summary
            print(
                self.name,
                "Ep:", global_episodes,
                "| Ep_r: %i" % global_rewards[-1],
                "| Avg. Reward: %.5f" % np.mean(env.arr_rewards),
                "| Min. Reward: %.5f" % np.min(env.arr_rewards),
                "| Max. Reward: %.5f" % np.max(env.arr_rewards),
                "| Avg. Timeheadway: %.5f" % np.mean(env.arr_th),
            )

            global_episodes += 1

            # store eps with crashes
            if crash == 1:
                arr_j = []
                if not os.path.exists(args.fpath + args.log_dir + '/results'):
                    os.makedirs(args.fpath + args.log_dir + '/results')
                # calculate jerk array
                for k in range(0, 5):
                    arr_j.append(float(0))

                for k in range(5, len(env.arr_t)):
                    # calculate vehicle jerk
                    if abs(env.arr_t[k] - env.arr_t[k - 5]) != 0:
                        arr_j.append(((env.arr_a[k]) - (env.arr_a[k - 5])) / (env.arr_t[k] - env.arr_t[k - 5]))  # jerk
                    else:
                        arr_j.append(0)

                # write results to file
                headers = ['t', 'j', 'v', 'a', 'v_lead', 'a_lead', 'x_rel', 'v_rel', 'th', 'y_0', 'y_sc', 'sc', 'cof']
                with open(args.fpath + args.log_dir + '/results/' + str(global_episodes) + '.csv', 'w', newline='\n') as f:
                    wr = csv.writer(f, delimiter=',')
                    rows = zip(env.arr_t, arr_j, env.arr_v, env.arr_a, env.arr_v_leader, env.arr_a_leader, env.arr_x,
                               env.arr_dv, env.arr_th, env.arr_y_0, env.arr_y_sc, env.arr_sc, env.arr_cof)
                    wr.writerow(headers)
                    wr.writerows(rows)

        print('Number of crashes: %d' % crash_count)


if __name__ == "__main__":
    global_rewards = []
    global_episodes = 0

    args = get_arguments()      # get arguments

    a2c_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=a2c_graph, config=config)

    with a2c_graph.as_default():
        global_ac = ACNet(args, GLOBAL_NET_SCOPE, sess)  # we only need its params
        worker = Worker(args, str('W_0'), global_ac, sess)


    # tensorboard summaries
    tf.summary.scalar('loss/policy_loss', worker.AC.a_loss)
    tf.summary.scalar('loss/value_loss', worker.AC.c_loss)
    tf.summary.histogram('mu', worker.AC.mu)
    tf.summary.histogram('sigma', worker.AC.sigma)
    tf.summary.histogram('v', worker.AC.v)
    tf.summary.histogram('v_target', worker.AC.v_target)
    tf.summary.histogram('act_out', worker.AC.A)

    with sess.as_default():
        with a2c_graph.as_default():
            saver = tf.train.Saver()
            tf.global_variables_initializer().run()

            # merge tensorboard summaries
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(args.fpath + args.log_dir, sess.graph)

    # run A2C algorithm
    worker.work()

    # save weights
    if not os.path.exists(args.fpath + args.log_dir):
        os.makedirs(args.fpath + args.log_dir)
    checkpoint_path = os.path.join(args.fpath + args.log_dir, "model-ep-%d-finalr-%d.ckpt" % (global_episodes, global_rewards[-1]))
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)