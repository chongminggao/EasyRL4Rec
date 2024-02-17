
import os
import sys
import numpy as np

from src.core.envs.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from src.core.envs.MovieLens.MovieLensData import MovieLensData

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/MovieLens"
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")



class MovieLensEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, lbe_user=None, lbe_item=None, mat_distance=None,
                 num_leave_compute=5, leave_threshold=80, max_turn=100, random_init=False):

        #self.max_turn = max_turn

        if mat is not None:
            self.mat = mat
            self.lbe_user = lbe_user
            self.lbe_item = lbe_item
            self.mat_distance = mat_distance
        else:
            self.mat, self.lbe_user, self.lbe_item, self.mat_distance = self.load_env_data()

        super(MovieLensEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def load_env_data():
        mat = MovieLensData.load_mat()
        lbe_user, lbe_item = MovieLensData.get_lbe()
        #print(mat.shape)
        mat_distance = MovieLensData.get_saved_distance_mat(mat, PRODATAPATH)
        #print(mat.shape)
        
        # np.percentile(mat_distance, 30)
        
        return mat, lbe_user, lbe_item, mat_distance
    
    def _determine_whether_to_leave(self, t, action):
        if t == 0:
            return False

        # # for debug:
        # np.percentile(self.mat_distance, [0,10,25,50,75,90,100])
        # res = [  0.        ,  53.60730985,  72.51212281, 100.98297015,
        #        142.92329426, 219.51983722, 810.23231568]
        # np.percentile(self.mat_distance, 0.33)
        # np.percentile(self.mat_distance, 0.34)
        # np.percentile(self.mat_distance, 30) # res = 78.04005882754639
        #
        # from matplotlib import pyplot as plt
        # plt.hist(self.mat_distance.reshape(-1), bins=100, alpha=0.5, color='blue', edgecolor='black')
        # plt.show()

        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        dist_list = np.array([self.mat_distance[action, x] for x in window_actions])
        if any(dist_list < self.leave_threshold):
            return True

        return False
