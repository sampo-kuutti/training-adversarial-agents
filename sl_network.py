# Author: Sampo Kuutti
# Organisation: University of Surrey / Connected & Autonomous Vehicles Lab
# Email: s.j.kuutti@surrey.ac.uk
# sl_network initialises a pre-train supervised learning network for longitudinal vehicle control actions
import tensorflow as tf
import sl_model2
import os
import numpy as np

NUM_INPUTS = 3
MODEL_FILE = 'model-step-901000-val-0.0150463.ckpt'
DATA_DIR = './data/'
LOG_DIR = '.models/sl_models/'


class SupervisedNetwork(object):
    """implements the supervised learning model for estimating vehicle host actions"""

    def __init__(self):
        # set up tf session and model
        #args = get_arguments()
        sl_graph = tf.Graph()
        sl_config = tf.ConfigProto()
        sl_config.gpu_options.allow_growth = True
        with sl_graph.as_default():   # create a new graph and sess for ipg_proxy
            self.model = sl_model2.SupervisedModel()

        self.sess_sl = tf.Session(graph=sl_graph, config=sl_config)
        with self.sess_sl.as_default():
            with sl_graph.as_default():
                saver = tf.train.Saver()
                checkpoint_path = os.path.join(LOG_DIR, MODEL_FILE)
                saver.restore(self.sess_sl, checkpoint_path)
        print('sl_model: Restored model: %s' % MODEL_FILE)

    def inference(self, x):
        with self.sess_sl.as_default():
            x = np.reshape(x, (1, NUM_INPUTS))       # reshape to a valid shape for input to nn
            y = self.model.y.eval(feed_dict={self.model.x: x})      # output network prediction
        return y