# -*- coding: utf-8 -*-
import itertools
import json
import os
import pickle
from collections import Counter

import gymnasium as gym

from gym import spaces
from numba import njit
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix


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
DATAPATH = os.path.join(ROOTPATH, "data")


class KuaiRandEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, yname, mat=None, mat_distance=None, list_feat=None, num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):

        self.yname = yname

        if mat is not None:
            self.mat = mat
            self.list_feat = list_feat
            self.mat_distance = mat_distance
        else:
            self.mat, self.list_feat, self.mat_distance = self.load_mat(yname)
        # self.list_feat_small = list(map(lambda x: self.list_feat[x], self.lbe_video.classes_))

        # self.df_item_cat = self.df_item.filter(regex="^feat", axis=1)
        # self.list_feat, df_feat = KuaiRandEnv.load_category()

        super(KuaiRandEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

        

    @staticmethod
    def get_saved_mat(yname, mat):
        distance_mat_path = os.path.join(DATAPATH, f"distance_mat_{yname}.csv")
        if os.path.isfile(distance_mat_path):
            print("loading small distance matrix...")
            mat_distance = pickle.load(open(distance_mat_path, "rb"))
            print("loading completed.")
        else:
            num_item = mat.shape[1]
            distance = np.zeros([num_item, num_item])
            print("computing distance matrix for the first time...")
            # mat_distance = get_distance_mat1(mat, distance)
            mat_distance = get_distance_mat(mat)
            print(f"saving the distance matrix to {distance_mat_path}...")
            pickle.dump(mat_distance, open(distance_mat_path, 'wb'))
        return mat_distance


    @staticmethod
    def load_mat(yname, read_user_num=None):
        filename = ""
        if yname == "is_click":
            filename = "kuairand_is_click.csv"
        elif yname == "is_like":
            filename = "kuairand_is_like.csv"
        elif yname == "long_view":
            filename = "kuairand_long_view.csv"
        elif yname == "watch_ratio_normed":
            filename = "kuairand_watchratio.csv"

        filepath_GT = os.path.join(ROOTPATH, "MF_results_GT", filename)
        df_mat = pd.read_csv(filepath_GT, header=0)

        if not read_user_num is None:
            df_mat_part = df_mat.loc[df_mat["user_id"] < read_user_num]
        else:
            df_mat_part = df_mat

        num_user = df_mat_part['user_id'].nunique()
        num_item = df_mat_part['item_id'].nunique()
        assert num_user == 27285
        assert num_item == 7583

        mat = csr_matrix((df_mat_part["value"], (df_mat_part['user_id'], df_mat_part['item_id'])),
                         shape=(num_user, num_item)).toarray()

        mat_distance = KuaiRandEnv.get_saved_mat(yname, mat)
        # mat_distance = get_distance_mat1(mat, distance)
        list_feat, df_feat = KuaiRandEnv.load_category()
        # df_item = KuaiRandEnv.load_item_feat()

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

        return mat, list_feat, mat_distance

    @staticmethod
    def load_user_feat():
        print("load user features")
        filepath = os.path.join(DATAPATH, 'user_features_pure.csv')
        df_user = pd.read_csv(filepath, usecols=['user_id', 'user_active_degree',
                                                 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                                                 'fans_user_num_range', 'friend_user_num_range',
                                                 'register_days_range'] + [f'onehot_feat{x}' for x in range(18)]
                              )
        for col in ['user_active_degree',
                    'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                    'fans_user_num_range', 'friend_user_num_range', 'register_days_range']:

            df_user[col] = df_user[col].map(lambda x: chr(0) if x == 'UNKNOWN' else x)
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1
        for col in [f'onehot_feat{x}' for x in range(18)]:
            df_user.loc[df_user[col].isna(), col] = -124
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1

        df_user = df_user.set_index("user_id")
        return df_user

    @staticmethod
    def get_df_kuairand(name, is_sort=True):
        filename = os.path.join(DATAPATH, name)
        df_data = pd.read_csv(filename,
                              usecols=['user_id', 'item_id', 'time_ms', 'is_like', 'is_click', 'long_view',
                                       'duration_normed', "watch_ratio_normed"])

        # df_data['watch_ratio'] = df_data["play_time_ms"] / df_data["duration_ms"]
        # df_data.loc[df_data['watch_ratio'].isin([np.inf, np.nan]), 'watch_ratio'] = 0
        # df_data.loc[df_data['watch_ratio'] > 5, 'watch_ratio'] = 5
        # df_data['duration_01'] = df_data['duration_ms'] / 1e5
        # df_data.rename(columns={"time_ms": "timestamp"}, inplace=True)
        # df_data["timestamp"] /= 1e3

        # load feature info
        list_feat, df_feat = KuaiRandEnv.load_category()
        df_data = df_data.join(df_feat, on=['item_id'], how="left")

        df_item = KuaiRandEnv.load_item_feat()

        # load user info
        df_user = KuaiRandEnv.load_user_feat()
        df_data = df_data.join(df_user, on=['user_id'], how="left")

        # get user sequences
        if is_sort:
            df_data.sort_values(["user_id", "time_ms"], inplace=True)
            df_data.reset_index(drop=True, inplace=True)

        return df_data, df_user, df_item, list_feat


    @staticmethod
    def get_domination():
        df_data, _, df_item, _ = KuaiRandEnv.get_df_kuairand("train_processed.csv")
        CODEDIRPATH = os.path.dirname(__file__)
        feature_domination_path = os.path.join(CODEDIRPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = get_sorted_domination_features(
                df_data, df_item, is_multi_hot=True, yname="is_click", threshold=1)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination

    @staticmethod
    def load_category():
        print("load item feature")
        filepath = os.path.join(DATAPATH, 'video_features_basic_pure.csv')
        df_item = pd.read_csv(filepath, usecols=["tag"], dtype=str)
        ind = df_item['tag'].isna()
        df_item['tag'].loc[~ind] = df_item['tag'].loc[~ind].map(lambda x: eval(f"[{x}]"))
        df_item['tag'].loc[ind] = [[-1]] * ind.sum()

        list_feat = df_item['tag'].to_list()

        df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2'])
        df_feat.index.name = "item_id"
        df_feat[df_feat.isna()] = -1
        df_feat = df_feat + 1
        df_feat = df_feat.astype(int)

        return list_feat, df_feat

    @staticmethod
    def load_item_feat(only_small=False):
        list_feat, df_feat = KuaiRandEnv.load_category()
        video_mean_duration = KuaiRandEnv.load_video_duration()
        df_item = df_feat.join(video_mean_duration, on=['item_id'], how="left")

        return df_item

    @staticmethod
    def load_video_duration():
        duration_path = os.path.join(DATAPATH, "video_duration_normed.csv")
        if os.path.isfile(duration_path):
            video_mean_duration = pd.read_csv(duration_path, header=0)["duration_normed"]
        else:
            small_path = os.path.join(DATAPATH, "test_processed.csv")
            small_duration = pd.read_csv(small_path, header=0, usecols=["item_id", 'duration_normed'])
            big_path = os.path.join(DATAPATH, "train_processed.csv")
            big_duration = pd.read_csv(big_path, header=0, usecols=["item_id", 'duration_normed'])
            duration_all = small_duration.append(big_duration)
            video_mean_duration = duration_all.groupby("item_id").agg(lambda x: sum(list(x)) / len(x))[
                "duration_normed"]
            video_mean_duration.to_csv(duration_path, index=False)

        video_mean_duration.index.name = "item_id"
        return video_mean_duration


    

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


@njit
def get_distance_mat1(mat, distance):
    matt = np.transpose(mat)
    for item_i in range(len(distance)):
        vec_i = matt[item_i]
        for item_j in range(len(distance)):
            vec_j = matt[item_j]
            dist = ((vec_i-vec_j)**2).sum()**0.5
            distance[item_i, item_j] = dist
    return distance

def get_distance_mat(mat):
    num_item = mat.shape[1]
    distance = np.zeros([num_item, num_item])
    for item_i in tqdm(range(len(distance))):
        vec_i = mat[:, item_i]
        a = vec_i - mat.T
        b = np.linalg.norm(a, axis=1)
        distance[item_i] = b
        # for item_j in range(len(distance)):
        #     vec_j = mat[:, item_j]
        #     dist = np.linalg.norm(vec_i-vec_j)
        #     distance[item_i, item_j] = dist
    return distance

    # import seaborn as sns
    # sns.histplot(c)
    # from matplotlib import pyplot as plt
    # plt.show()

