import os
import sys
import numpy as np

from environments.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from environments.YahooR3.YahooData import YahooData

ROOTPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOTPATH, "data_raw")


class YahooEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, mat_distance=None,
                 num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):
        if mat is not None:
            self.mat = mat
            self.mat_distance = mat_distance
        else:
            self.mat, self.mat_distance = self.load_mat()
        # smallmat shape: (1411, 3327)
        super(YahooEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)
        
    @staticmethod
    def load_env_data():
        mat = YahooData.load_mat()
        mat_distance = YahooData.get_saved_distance_mat(mat)
        mat = mat[:5400,:]
        return mat, mat_distance
    
    def _determine_whether_to_leave(self, t, action):
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        dist_list = np.array([self.mat_distance[action, x] for x in window_actions])
        if any(dist_list < self.leave_threshold):
            return True

        return False
