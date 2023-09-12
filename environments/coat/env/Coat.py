# -*- coding: utf-8 -*-
# @Time    : 2022/9/15 22:40
# @Author  : Chongming GAO
# @FileName: Coat.py

import os
import pickle

import gymnasium as gym
import torch
from gym import spaces
from numba import njit


import pandas as pd
import numpy as np
import random

from tqdm import tqdm
import sys

from environments.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.util.utils import get_sorted_domination_features

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = ROOTPATH


class CoatEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, df_item=None, mat_distance=None, num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):

        if mat is not None:
            self.mat = mat
            self.df_item = df_item
            self.mat_distance = mat_distance
        else:
            self.mat, self.df_item, self.mat_distance = self.load_mat()

        super(CoatEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def get_df_coat(name):
        # read interaction
        filename = os.path.join(DATAPATH, name)
        mat_train = pd.read_csv(filename, sep="\s+", header=None)
        df_data = pd.DataFrame([], columns=["user_id", "item_id", "rating"])

        for item in mat_train.columns:
            one_item = mat_train.loc[mat_train[item] > 0, item].reset_index().rename(
                columns={"index": "user_id", item: "rating"})
            one_item["item_id"] = item
            df_data = pd.concat([df_data, one_item])
        df_data.reset_index(drop=True, inplace=True)

        # read user feature
        df_user = CoatEnv.load_user_feat()

        # read item features
        df_item = CoatEnv.load_item_feat()

        df_data = df_data.join(df_user, on="user_id", how='left')
        df_data = df_data.join(df_item, on="item_id", how='left')

        df_data = df_data.astype(int)
        list_feat = None

        return df_data, df_user, df_item, list_feat

    @staticmethod
    def get_domination():
        df_data, _, df_item, _ = CoatEnv.get_df_coat("train.ascii")
        CODEDIRPATH = os.path.dirname(__file__)
        feature_domination_path = os.path.join(CODEDIRPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = get_sorted_domination_features(
                df_data, df_item, is_multi_hot=False)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination

    @staticmethod
    def load_exposure_and_popularity(predicted_mat, filename="train.ascii"):
        filepath = os.path.join(DATAPATH, filename)
        mat_train = pd.read_csv(filepath, sep="\s+", header=None)
        isexposure = mat_train > 0
        df_popularity = isexposure.sum()

        df_train, _, _ = CoatEnv.get_df_coat("train.ascii")
        df_frequency = df_train.groupby(["user_id", "item_id"])["rating"].agg(len)

        return df_frequency, df_popularity

    @staticmethod
    def load_mat():
        filename_GT = os.path.join(DATAPATH, "..", "RL4Rec", "data", "coat_pseudoGT_ratingM.ascii")
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)

        # mat_distance = get_distance_mat(mat)

        num_item = mat.shape[1]
        distance = np.zeros([num_item, num_item])
        mat_distance = get_distance_mat1(mat, distance)

        df_item = CoatEnv.load_item_feat()

        # dist_cat = np.zeros_like(mat_distance)
        # for i in range(len(dist_cat)):
        #     for j in range(len(dist_cat)):
        #         sim = (sum(df_item.loc[i] - df_item.loc[j] == 0) / len(df_item.columns))
        #         dist_cat[i,j] = 6 if sim == 0 else 1 / sim
        #
        #
        # dist_cat[np.isinf(dist_cat)] = 6
        # dist_cat = dist_cat * 3
        # df = pd.DataFrame(zip(mat_distance.reshape([-1]), dist_cat.reshape([-1])), columns=["dist","category"])
        #
        # df.groupby("category").agg(np.mean)
        #
        # import seaborn as sns
        # sns.boxplot(x = df["category"], y=df["dist"])
        # from matplotlib import pyplot as plt
        # plt.show()

        # import seaborn as sns
        # sns.histplot(data=mat_distance.reshape([-1]))
        # from matplotlib import pyplot as plt
        # plt.show()

        return mat, df_item, mat_distance

    @staticmethod
    def load_user_feat():
        filename_user = os.path.join(DATAPATH, "user_item_features", "user_features.ascii")
        mat_user = pd.read_csv(filename_user, sep="\s+", header=None, dtype=str)

        feat_cols_user = [2, 6, 3, 3]
        feat_cols_user = [0] + list(np.cumsum(feat_cols_user))

        df_user = pd.DataFrame([], columns=['gender_u', 'age', 'location', 'fashioninterest'], dtype=int)
        for k, (left, right) in enumerate(zip(feat_cols_user[:-1], feat_cols_user[1:])):
            col = mat_user[range(left, right)].apply(lambda x: "".join(x), axis=1)
            col_int = col.map(lambda x: np.log2(int(x[::-1], 2)))
            assert not any(col_int % 1)
            df_user[df_user.columns[k]] = col_int.astype(dtype=int)
        df_user.index.name = "user_id"
        return df_user

    @staticmethod
    def load_item_feat():
        filename_item = os.path.join(DATAPATH, "user_item_features", "item_features.ascii")
        mat_item = pd.read_csv(filename_item, sep="\s+", header=None, dtype=str)

        feat_cols_item = [0, 2, 18, 31, 33]

        df_item = pd.DataFrame([], columns=['gender_i', 'jackettype', 'color', 'onfrontpage'], dtype=int)
        for k, (left, right) in enumerate(zip(feat_cols_item[:-1], feat_cols_item[1:])):
            col = mat_item[range(left, right)].apply(lambda x: "".join(x), axis=1)
            col_int = col.map(lambda x: np.log2(int(x[::-1], 2)))
            assert not any(col_int % 1)
            df_item[df_item.columns[k]] = col_int.astype(dtype=int)
        df_item.index.name = "item_id"

        return df_item

    @property
    def state(self):
        if self.action is None:
            res = self.cur_user
        else:
            res = self.action
        return np.array([res])

    def __user_generator(self):
        user = random.randint(0, len(self.mat) - 1)
        # # todo for debug
        # user = 0
        return user

    def step(self, action):
        # action = int(action)

        # Action: tensor with shape (32, )
        self.action = action
        t = self.total_turn
        terminated = self._determine_whether_to_leave(t, action)
        if t >= (self.max_turn - 1):
            terminated = True
        self._add_action_to_history(t, action)

        reward = self.mat[self.cur_user, action]

        self.cum_reward += reward
        self.total_turn += 1

        # if terminated:
        #     self.cur_user = self.__user_generator()

        return self.state, reward, terminated, False, {'cum_reward': self.cum_reward}

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.cur_user = self.__user_generator()

        self.action = None  # Add by Chongming
        self._reset_history()

        # return self.state, {'key': 1, 'env': self}
        return self.state, {'cum_reward': 0.0}

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

    def _reset_history(self):
        self.history_action = {}
        self.sequence_action = []
        self.max_history = 0

    def _add_action_to_history(self, t, action):

        self.sequence_action.append(action)
        self.history_action[t] = action

        assert self.max_history == t
        self.max_history += 1


@njit
def get_distance_mat1(mat, distance):
    matt = np.transpose(mat)
    for item_i in range(len(distance)):
        vec_i = matt[item_i]
        for item_j in range(len(distance)):
            vec_j = matt[item_j]
            dist = ((vec_i - vec_j) ** 2).sum() ** 0.5
            distance[item_i, item_j] = dist
    return distance


# @njit
def get_distance_mat(mat):
    num_item = mat.shape[1]
    distance = np.zeros([num_item, num_item])
    for item_i in tqdm(range(len(distance))):
        vec_i = mat[:, item_i]
        for item_j in range(len(distance)):
            vec_j = mat[:, item_j]
            dist = np.linalg.norm(vec_i - vec_j)
            distance[item_i, item_j] = dist

    return distance

    # import seaborn as sns
    # sns.histplot(data=distance.reshape([-1]))
    # from matplotlib import pyplot as plt
    # plt.show()


# def negative_sampling(df_train, df_item, df_user, y_name):
#     print("negative sampling...")
#     mat_train = csr_matrix((df_train[y_name], (df_train["user_id"], df_train["item_id"])),
#                            shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1)).toarray()
#     df_negative = np.zeros([len(df_train), 2])
#
#     user_ids = df_train["user_id"].to_numpy()
#     item_ids = df_train["item_id"].to_numpy()
#
#     find_negative(user_ids, item_ids, mat_train, df_negative, is_rand=True)
#     df_negative = pd.DataFrame(df_negative, columns=["user_id", "item_id"], dtype=int)
#
#     df_negative = df_negative.join(df_item, on=['item_id'], how="left")
#     df_negative = df_negative.join(df_user, on=['user_id'], how="left")
#
#     # df_negative.loc[df_negative["duration_ms"].isna(), "duration_ms"] = 0
#     return df_train, df_negative

def construct_complete_val_x(dataset_val, user_features, item_features):
    df_item = CoatEnv.load_item_feat()
    df_user = CoatEnv.load_user_feat()

    user_ids = np.arange(dataset_val.x_columns[dataset_val.user_col].vocabulary_size)
    item_ids = np.arange(dataset_val.x_columns[dataset_val.item_col].vocabulary_size)
    df_user_complete = pd.DataFrame(
        df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
        columns=df_user.reset_index()[user_features].columns)
    df_item_complete = pd.DataFrame(np.tile(df_item.reset_index()[item_features], (len(user_ids), 1)),
                                    columns=df_item.reset_index()[item_features].columns)

    # np.tile(np.concatenate([np.expand_dims(df_item_env.index.to_numpy(), df_item_env.to_numpy()], axis=1), (2, 1))

    df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)
    return df_x_complete


def compute_normed_reward_for_all(user_model, dataset_val, user_features, item_features):
    df_x_complete = construct_complete_val_x(dataset_val, user_features, item_features)
    n_user, n_item = df_x_complete[["user_id", "item_id"]].nunique()
    predict_mat = np.zeros((n_user, n_item))

    for i, user in tqdm(enumerate(range(n_user)), total=n_user, desc="predict all users' rewards on all items"):
        ui = torch.tensor(df_x_complete[df_x_complete["user_id"] == user].to_numpy(), dtype=torch.float,
                          device=user_model.device, requires_grad=False)
        reward_u = user_model.forward(ui).detach().squeeze().cpu().numpy()
        predict_mat[i] = reward_u

    minn = predict_mat.min()
    maxx = predict_mat.max()

    normed_mat = (predict_mat - minn) / (maxx - minn)

    return normed_mat
