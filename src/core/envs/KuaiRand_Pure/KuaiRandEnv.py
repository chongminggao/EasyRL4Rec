import os
import sys
import itertools
from collections import Counter

from src.core.envs.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from src.core.envs.KuaiRand_Pure.KuaiRandData import KuaiRandData

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/KuaiRand_Pure"
DATAPATH = os.path.join(ROOTPATH, "data_raw")


class KuaiRandEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, yname, mat=None, mat_distance=None, list_feat=None, 
                 num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):
        self.yname = yname
        if mat is not None:
            self.mat = mat
            self.list_feat = list_feat
            self.mat_distance = mat_distance
        else:
            self.mat, self.list_feat, self.mat_distance = self.load_env_data(yname)

        super(KuaiRandEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def load_env_data(yname="is_click", read_user_num=None):
        mat = KuaiRandData.load_mat(yname, read_user_num)
        list_feat, df_feat = KuaiRandData.load_category()
        mat_distance = KuaiRandData.get_saved_distance_mat(yname, mat)
        return mat, list_feat, mat_distance
   
    def _determine_whether_to_leave(self, t, action):
        # self.list_feat[action]
        if t == 0:
            return False

        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        hist_categories_each = list(map(lambda x: self.list_feat[x], window_actions))

        hist_categories = list(itertools.chain(*hist_categories_each))
        hist_dict = Counter(hist_categories)
        category_a = self.list_feat[action]
        for c in category_a:
            if hist_dict[c] > self.leave_threshold:
                return True

        # window_actions = self.sequence_action[t - self.num_leave_compute:t]
        # dist_list = np.array([self.mat_distance[action, x] for x in window_actions])
        # if any(dist_list < self.leave_threshold):
        #     return True

        return False

