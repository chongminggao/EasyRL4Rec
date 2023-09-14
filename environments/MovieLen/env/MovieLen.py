
# -*- coding: utf-8 -*-
# @Time    : 2023/8/24
# @Author  : Yuefeng Sun
# @FileName: MovieLen.py  movielens-1m

import os

import gym
import torch
from gym import spaces
from numba import njit
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from collections import Counter
import itertools

import pandas as pd
import numpy as np
import random

from tqdm import tqdm

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = ROOTPATH




class MovielenEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, mat_distance=None,
                 num_leave_compute=5, leave_threshold=1, max_turn=100):

        self.max_turn = max_turn

        if mat is not None:
            self.mat = mat
            self.mat_distance = mat_distance
        else:
            self.mat, self.mat_distance = self.load_mat()

        # smallmat shape: (1411, 3327)

        self.observation_space = spaces.Box(low=0, high=len(self.mat) - 1, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Box(low=0, high=self.mat.shape[1] - 1, shape=(1,), dtype=np.int32)

        self.num_leave_compute = num_leave_compute
        self.leave_threshold = leave_threshold

        self.reset()

    @staticmethod
    def get_df_movielen(name):
        # read interaction
        filename = os.path.join(DATAPATH, name)
        #df_data = pd.read_csv(filename, sep="\s+", header=None, names=["user_id", "item_id", "rating"])
        df_data = pd.read_csv(filename,  header=0, names=["user_id", "item_id", "rating"])

        df_data["user_id"] -= 1
        df_data["item_id"] -= 1

        df_user = MovielenEnv.load_user_feat()
        df_item = MovielenEnv.load_item_feat()
        list_feat = None

        return df_data, df_user, df_item, list_feat

    @staticmethod
    def load_user_feat():
        df_user = pd.DataFrame(np.arange(6040), columns=["user_id"])   #exchange user&item
        df_user.set_index("user_id", inplace=True)
        return df_user

    @staticmethod
    def load_item_feat():
        df_item = pd.DataFrame(np.arange(3952), columns=["item_id"])
        df_item.set_index("item_id", inplace=True)
        return df_item


    @staticmethod
    def load_mat():
        #filename_GT = os.path.join(DATAPATH, "..", "RL4Rec", "data", "yahoo_pseudoGT_ratingM.ascii")
        filename_GT = os.path.join(DATAPATH, "movielen.csv")
        mat = pd.read_csv(filename_GT, header=0).to_numpy(dtype=int)
        mat = mat.T

        num_item = mat.shape[1]
        distance = np.zeros([num_item, num_item])
        mat_distance = get_distance_mat(mat,distance)

        # import seaborn as sns
        # sns.histplot(data=mat_distance.reshape([-1]))
        # from matplotlib import pyplot as plt
        # plt.show()

        #mat = mat[:1983,:]


        return mat, mat_distance


    @staticmethod
    def compute_normed_reward(user_model, lbe_user, lbe_video, df_video_env):
        # filename = "normed_reward.pickle"
        # filepath = os.path.join(DATAPATH, filename)

        # if os.path.isfile(filepath):
        #     with open(filepath, "rb") as file:
        #         normed_mat = pickle.load(file)
        #     return normed_mat

        n_user = len(lbe_user.classes_)
        n_item = len(lbe_video.classes_)

        item_info = df_video_env.loc[lbe_video.classes_]
        item_info["item_id"] = item_info.index
        item_info = item_info[["item_id", "feat0", "feat1", "feat2", "feat3", "video_duration"]]
        item_np = item_info.to_numpy()

        predict_mat = np.zeros((n_user, n_item))

        for i, user in tqdm(enumerate(lbe_user.classes_), total=n_user, desc="predict all users' rewards on all items"):
            ui = torch.tensor(np.concatenate((np.ones((n_item, 1)) * user, item_np), axis=1),
                              dtype=torch.float, device=user_model.device, requires_grad=False)
            reward_u = user_model.forward(ui).detach().squeeze().cpu().numpy()
            predict_mat[i] = reward_u

        minn = predict_mat.min()
        maxx = predict_mat.max()

        normed_mat = (predict_mat - minn) / (maxx - minn)

        return normed_mat

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
        done = self._determine_whether_to_leave(t, action)
        if t >= (self.max_turn - 1):
            done = True
        self._add_action_to_history(t, action)

        reward = self.mat[self.cur_user, action]

        self.cum_reward += reward
        self.total_turn += 1

        # if done:
        #     self.cur_user = self.__user_generator()

        return self.state, reward, done, {'cum_reward': self.cum_reward}

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.cur_user = self.__user_generator()

        self.action = None  # Add by Chongming
        self._reset_history()

        return self.state

    def render(self, mode='human', close=False):
        # history_action = self.history_action
        # category = {k: self.list_feat_small[v] for k, v in history_action.items()}
        # category_debug = {k:self.list_feat[v] for k,v in history_action.items()}
        # return history_action, category, category_debug
        # return self.cur_user, history_action, category
        pass

    def _determine_whether_to_leave(self, t, action):
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        dist_list = np.array([self.mat_distance[action, x] for x in window_actions])
        if any(dist_list < self.leave_threshold):
            return True

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


# @njit
# def find_negative(user_ids, item_ids, mat_train, df_negative, K=1):
#     for i in range(len(user_ids)):
#         user, item = user_ids[i], item_ids[i]
#         value = mat_train[user, item]
#
#         neg = item + 1
#         # neg_v = mat_train[user, neg]
#
#         while neg < mat_train.shape[1]:
#             neg_v = mat_train[user, neg]
#             if neg_v > 0:
#                 neg += 1
#             else:
#                 df_negative[i, 0] = user
#                 df_negative[i, 1] = neg
#                 break
#         else:
#             neg = item - 1
#             while neg >= 0:
#                 neg_v = mat_train[user, neg]
#                 if neg_v > 0:
#                     neg -= 1
#                 else:
#                     df_negative[i, 0] = user
#                     df_negative[i, 1] = neg
#                     break


@njit
def get_distance_mat(mat, distance):
    matt = np.transpose(mat)
    for item_i in range(len(distance)):
        vec_i = matt[item_i]
        for item_j in range(len(distance)):
            vec_j = matt[item_j]
            dist = ((vec_i-vec_j)**2).sum()**0.5
            distance[item_i, item_j] = dist
    return distance

def negative_sampling(df_train, df_user, df_item, y_name):
    print("negative sampling...")
    mat_train = csr_matrix((df_train[y_name], (df_train["user_id"], df_train["item_id"])),
                           shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1)).toarray()
    df_negative = np.zeros([len(df_train), 2])

    user_ids = df_train["user_id"].to_numpy()
    item_ids = df_train["item_id"].to_numpy()

    find_negative(user_ids, item_ids, mat_train, df_negative)
    df_negative = pd.DataFrame(df_negative, columns=["user_id", "item_id"], dtype=int)

    # df_negative.loc[df_negative["duration_ms"].isna(), "duration_ms"] = 0
    return df_train, df_negative


def construct_complete_val_x(dataset_val, user_features, item_features):

    user_ids = np.arange(dataset_val.x_columns[dataset_val.user_col].vocabulary_size)
    # user_ids = np.arange(1000)
    item_ids = np.arange(dataset_val.x_columns[dataset_val.item_col].vocabulary_size)

    df_item = dataset_val.df_item_val
    #df_user = pd.DataFrame(range(15400), columns=["user_id"])
    df_user = pd.DataFrame(range(1683), columns=["user_id"])

    # df_user_complete = pd.DataFrame({"user_id": user_ids.repeat(len(item_ids))})
    df_user_complete = pd.DataFrame(
        df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
        columns=df_user.reset_index()[user_features].columns)
    df_item_complete = pd.DataFrame(np.tile(df_item.reset_index()[item_features], (len(user_ids), 1)),
                                    columns=df_item.reset_index()[item_features].columns)

    df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)
    return df_x_complete

def compute_normed_reward_for_all(user_model, dataset_val, user_features, item_features):
    df_x_complete = construct_complete_val_x(dataset_val, user_features, item_features)
    n_user, n_item = df_x_complete[["user_id", "item_id"]].nunique()
    predict_mat = np.zeros((n_user, n_item))

    for i, user in tqdm(enumerate(range(n_user)), total=n_user, desc="predict all users' rewards on all items"):
        ui = torch.tensor(df_x_complete[df_x_complete["user_id"] == user].to_numpy(),dtype=torch.float, device=user_model.device, requires_grad=False)
        reward_u = user_model.forward(ui).detach().squeeze().cpu().numpy()
        predict_mat[i] = reward_u

    minn = predict_mat.min()
    maxx = predict_mat.max()

    normed_mat = (predict_mat - minn) / (maxx - minn)

    return normed_mat
