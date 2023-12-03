import os
import sys
import numpy as np

from environments.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from environments.coat.CoatData import CoatData

ROOTPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")


class CoatEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, df_item=None, mat_distance=None, 
                 num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):
        if mat is not None:
            self.mat = mat
            self.df_item = df_item
            self.mat_distance = mat_distance
        else:
            self.mat, self.df_item, self.mat_distance = self.load_env_data()

        super(CoatEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def load_env_data():
        mat = CoatData.load_mat()
        df_item = CoatData.load_item_feat()
        mat_distance = CoatData.get_saved_distance_mat(mat, PRODATAPATH)
        return mat, df_item, mat_distance

    def _determine_whether_to_leave(self, t, action):
        # self.list_feat[action]
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]

        dist_list = np.array([self.mat_distance[action, x] for x in window_actions])

        if any(dist_list < self.leave_threshold):
            return True

        # hist_categories_each = list(map(lambda x: self.list_feat_small[x], window_actions))
        # hist_set = set.union(*list(map(lambda x: self.list_feat[x], self.sequence_action[t - self.num_leave_compute:t-1])))
        # hist_categories = list(itertools.chain(*hist_categories_each))
        # hist_dict = Counter(hist_categories)
        # category_a = self.list_feat_small[action]

        # for c in category_a:
        #     if hist_dict[c] > self.leave_threshold:
        #         return True

        # if action in window_actions:
        #     return True

        return False
