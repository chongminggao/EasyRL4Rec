import os
import sys
import itertools
from collections import Counter

from environments.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from environments.KuaiRec.KuaiData import KuaiData

ROOTPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOTPATH, "data_raw")


class KuaiEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, lbe_user=None, lbe_item=None, list_feat=None, df_dist_small=None,
                 num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):
        if mat is not None:
            self.mat = mat
            self.lbe_user = lbe_user
            self.lbe_item = lbe_item
            self.list_feat = list_feat
            self.df_dist_small = df_dist_small
        else:
            self.mat, self.lbe_user, self.lbe_item, self.list_feat, self.df_dist_small = self.load_env_data()

        self.list_feat_small = list(map(lambda x: self.list_feat[x], self.lbe_item.classes_))
        # smallmat shape: (1411, 3327)
        super(KuaiEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def load_env_data():
        mat, lbe_user, lbe_item = KuaiData.load_mat()
        list_feat, df_feat = KuaiData.load_category()
        df_dist_small = KuaiData.get_saved_distance_mat(list_feat, lbe_item.classes_, DATAPATH)
        return mat, lbe_user, lbe_item, list_feat, df_dist_small

    def render(self, mode='human', close=False):
        history_action = self.history_action
        category = {k: self.list_feat_small[v] for k, v in history_action.items()}
        # category_debug = {k:self.list_feat[v] for k,v in history_action.items()}
        # return history_action, category, category_debug
        return self.cur_user, history_action, category

    def _determine_whether_to_leave(self, t, action):
        # self.list_feat[action]
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        hist_categories_each = list(map(lambda x: self.list_feat_small[x], window_actions))

        # hist_set = set.union(*list(map(lambda x: self.list_feat[x], self.sequence_action[t - self.num_leave_compute:t-1])))

        hist_categories = list(itertools.chain(*hist_categories_each))
        hist_dict = Counter(hist_categories)
        category_a = self.list_feat_small[action]
        for c in category_a:
            if hist_dict[c] > self.leave_threshold:
                return True

        # if action in window_actions:
        #     return True

        return False
    