# Author: Sampo Kuutti
# Organisation: University of Surrey / Connected & Autonomous Vehicles Lab
# Email: s.j.kuutti@surrey.ac.uk
# ipg_proxy is a learned model used to speed up RL training by using the model as proxy for the simulator
# Uses a feed-forward network to estimate longitudinal vehicle dynamics based on IPG CarMaker data
import sl_model
import tensorflow as tf
import numpy as np
import os

fpath = '/vol/research/safeav/Sampo/condor-a2c/test/'  # use same directory path
NUM_INPUTS = 5
MODEL_FILE = 'model-step-981000-val-7.87466e-05.ckpt'
DATA_DIR = './data/'
LOG_DIR = fpath


class IpgProxy(object):
    """implements the ipg proxy model for emulating the IPG CarMaker simulation environment
    """

    def __init__(self):
        # set up tf session and model
        ipg_graph = tf.Graph()
        with ipg_graph.as_default():   # create a new graph and sess for ipg_proxy
            self.model = sl_model.SupervisedModel()
        self.sess_1 = tf.Session(graph=ipg_graph)
        with self.sess_1.as_default():
            with ipg_graph.as_default():
                saver = tf.train.Saver()
                checkpoint_path = os.path.join(LOG_DIR, MODEL_FILE)
                saver.restore(self.sess_1, checkpoint_path)
        print('ipg_proxy: Restored model: %s' % MODEL_FILE)

    def inference(self, x):
        with self.sess_1.as_default():
            x = np.reshape(x, (1, NUM_INPUTS))       # reshape to a valid shape for input to nn
            y = self.model.y.eval(feed_dict={self.model.x: x})      # output network prediction
        return y