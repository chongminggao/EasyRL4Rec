# -*- coding: utf-8 -*-
# @Time    : 2021/10/1 3:03 下午
# @Author  : Chongming GAO
# @FileName: kuaiEnv.py
import json
import os
import pickle

import gymnasium as gym
import torch
# from gym import spaces
from numba import njit
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from collections import Counter
import itertools

import pandas as pd
import numpy as np
import random

from tqdm import tqdm

import sys

from environments.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.util.utils import get_sorted_domination_features

# from core.util.utils import get_similarity_mat, get_distance_mat

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = os.path.join(ROOTPATH, "data")


class KuaiEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, lbe_user=None, lbe_item=None, list_feat=None, df_video_env=None, df_dist_small=None,
                 num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):

        if mat is not None:
            self.mat = mat
            self.lbe_user = lbe_user
            self.lbe_item = lbe_item
            self.list_feat = list_feat
            self.df_video_env = df_video_env
            self.df_dist_small = df_dist_small
        else:
            self.mat, self.lbe_user, self.lbe_item, self.list_feat, self.df_video_env, self.df_dist_small = self.load_mat()

        self.list_feat_small = list(map(lambda x: self.list_feat[x], self.lbe_item.classes_))

        # smallmat shape: (1411, 3327)

        super(KuaiEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def get_df_kuairec(name="big_matrix_processed.csv"):
        filename = os.path.join(DATAPATH, name)
        df_data = pd.read_csv(filename,
                              usecols=['user_id', 'item_id', 'timestamp', 'watch_ratio_normed', 'duration_normed'])

        # df_data['duration_normed'] = df_data['duration_ms'] / 1000

        # load feature info
        list_feat, df_feat = KuaiEnv.load_category()

        if name == "big_matrix_processed.csv":
            only_small = False
        else:
            only_small = True
        df_user = KuaiEnv.load_user_feat(only_small)
        df_item = KuaiEnv.load_item_feat(only_small)

        df_data = df_data.join(df_feat, on=['item_id'], how="left")

        # if is_require_feature_domination:
        #     item_feat_domination = KuaiEnv.get_domination(df_data, df_item)
        #     return df_data, df_user, df_item, list_feat, item_feat_domination

        return df_data, df_user, df_item, list_feat

    @staticmethod
    def get_domination():
        df_data, _, df_item, _ = KuaiEnv.get_df_kuairec("big_matrix_processed.csv")
        CODEDIRPATH = os.path.dirname(__file__)
        feature_domination_path = os.path.join(CODEDIRPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            # item_feat_domination = get_sorted_domination_features(
            #     df_data, df_item, is_multi_hot=True, yname="watch_ratio_normed", threshold=0.6)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination
    
    @staticmethod
    def get_item_similarity(manner="tag"):
        
        CODEDIRPATH = os.path.dirname(__file__)
        item_similarity_path = os.path.join(CODEDIRPATH, "item_similarity.pickle")

        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            list_feat, df_feat = KuaiEnv.load_category()
            item_similarity = get_similarity_mat(list_feat, DATAPATH)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
        
    @staticmethod
    def get_item_popularity():
        # df_data, df_user, df_item, list_feat = KuaiEnv.get_df_kuairec("big_matrix_processed.csv")
        CODEDIRPATH = os.path.dirname(__file__)
        item_popularity_path = os.path.join(CODEDIRPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            filename = os.path.join(DATAPATH, "big_matrix_processed.csv")
            df_data = pd.read_csv(filename, usecols=['user_id', 'item_id', 'timestamp', 'watch_ratio'])
            n_users = df_data['user_id'].nunique()
            n_items = df_data['item_id'].nunique()
            
            df_data_filtered = df_data[df_data['watch_ratio']>=1.]
            
            groupby = df_data_filtered.loc[:, ["user_id", "item_id"]].groupby(by="item_id")
            df_pop = groupby.user_id.apply(list).reset_index()
            df_pop["popularity"] = df_pop['user_id'].apply(lambda x: len(x) / n_users)

            item_pop_df = pd.DataFrame(np.arange(n_items), columns=["item_id"])
            item_pop_df = item_pop_df.merge(df_pop, how="left", on="item_id")
            item_pop_df['popularity'].fillna(0, inplace=True)
            item_popularity = item_pop_df['popularity']
            pickle.dump(item_popularity, open(item_popularity_path, 'wb'))
        return item_popularity

    @staticmethod
    def load_user_feat(only_small=False):
        print("load user features")
        filepath = os.path.join(DATAPATH, 'user_features.csv')
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

        if only_small:
            lbe_user, lbe_item = KuaiEnv.get_lbe()
            user_list = lbe_user.classes_
            df_user_env = df_user.loc[user_list]
            return df_user_env

        return df_user

    @staticmethod
    def load_item_feat(only_small=False):
        list_feat, df_feat = KuaiEnv.load_category()
        video_mean_duration = KuaiEnv.load_video_duration()
        df_item = df_feat.join(video_mean_duration, on=['item_id'], how="left")

        if only_small:
            lbe_user, lbe_item = KuaiEnv.get_lbe()
            item_list = lbe_item.classes_
            df_item_env = df_item.loc[item_list]
            return df_item_env

        return df_item

    @staticmethod
    def get_lbe():
        if not os.path.isfile(os.path.join(DATAPATH, "user_id_small.csv")) or not os.path.isfile(
                os.path.join(DATAPATH, "item_id_small.csv")):
            small_path = os.path.join(DATAPATH, "small_matrix_processed.csv")
            df_small = pd.read_csv(small_path, header=0, usecols=['user_id', 'item_id'])

            user_id_small = pd.DataFrame(df_small["user_id"].unique(), columns=["user_id_small"])
            item_id_small = pd.DataFrame(df_small["item_id"].unique(), columns=["item_id_small"])

            user_id_small.to_csv(os.path.join(DATAPATH, "user_id_small.csv"), index=False)
            item_id_small.to_csv(os.path.join(DATAPATH, "item_id_small.csv"), index=False)
        else:
            user_id_small = pd.read_csv(os.path.join(DATAPATH, "user_id_small.csv"))
            item_id_small = pd.read_csv(os.path.join(DATAPATH, "item_id_small.csv"))

        lbe_user = LabelEncoder()
        lbe_user.fit(user_id_small["user_id_small"])

        lbe_item = LabelEncoder()
        lbe_item.fit(item_id_small["item_id_small"])

        return lbe_user, lbe_item

    @staticmethod
    def load_category():
        # load categories:
        print("load item feature")
        filepath = os.path.join(DATAPATH, 'item_categories.csv')
        df_feat0 = pd.read_csv(filepath, header=0)
        df_feat0.feat = df_feat0.feat.map(eval)

        list_feat = df_feat0.feat.to_list()
        # df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3'], dtype=int)
        df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3'])
        df_feat.index.name = "item_id"
        df_feat[df_feat.isna()] = -1
        df_feat = df_feat + 1
        df_feat = df_feat.astype(int)

        return list_feat, df_feat

    @staticmethod
    def load_video_duration():
        duration_path = os.path.join(DATAPATH, "video_duration_normed.csv")
        if os.path.isfile(duration_path):
            video_mean_duration = pd.read_csv(duration_path, header=0)["duration_normed"]
        else:
            small_path = os.path.join(DATAPATH, "small_matrix_processed.csv")
            small_duration = pd.read_csv(small_path, header=0, usecols=["item_id", 'duration_normed'])
            big_path = os.path.join(DATAPATH, "big_matrix_processed.csv")
            big_duration = pd.read_csv(big_path, header=0, usecols=["item_id", 'duration_normed'])
            duration_all = small_duration.append(big_duration)
            video_mean_duration = duration_all.groupby("item_id").agg(lambda x: sum(list(x)) / len(x))[
                "duration_normed"]
            video_mean_duration.to_csv(duration_path, index=False)

        video_mean_duration.index.name = "item_id"
        return video_mean_duration

    @staticmethod
    def load_mat():
        small_path = os.path.join(DATAPATH, "small_matrix_processed.csv")
        df_small = pd.read_csv(small_path, header=0, usecols=['user_id', 'item_id', 'watch_ratio'])
        # df_small['watch_ratio'][df_small['watch_ratio'] > 5] = 5
        df_small.loc[df_small['watch_ratio'] > 5, 'watch_ratio'] = 5

        lbe_item = LabelEncoder()
        lbe_item.fit(df_small['item_id'].unique())

        lbe_user = LabelEncoder()
        lbe_user.fit(df_small['user_id'].unique())

        mat = csr_matrix(
            (df_small['watch_ratio'],
             (lbe_user.transform(df_small['user_id']), lbe_item.transform(df_small['item_id']))),
            shape=(df_small['user_id'].nunique(), df_small['item_id'].nunique())).toarray()

        mat[np.isnan(mat)] = df_small['watch_ratio'].mean()
        mat[np.isinf(mat)] = df_small['watch_ratio'].mean()

        # load feature info
        list_feat, df_feat = KuaiEnv.load_category()

        # Compute the average video duration
        video_mean_duration = KuaiEnv.load_video_duration()

        video_list = df_small['item_id'].unique()
        df_video_env = df_feat.loc[video_list]
        df_video_env['duration_normed'] = np.array(
            list(map(lambda x: video_mean_duration[x], df_video_env.index)))

        # load or construct the distance mat (between item pairs):
        df_dist_small = get_distance_mat(list_feat, lbe_item.classes_, DATAPATH)

        return mat, lbe_user, lbe_item, list_feat, df_video_env, df_dist_small

    @staticmethod
    def compute_normed_reward(user_model, lbe_user, lbe_item, df_video_env):
        # filename = "normed_reward.pickle"
        # filepath = os.path.join(DATAPATH, filename)

        # if os.path.isfile(filepath):
        #     with open(filepath, "rb") as file:
        #         normed_mat = pickle.load(file)
        #     return normed_mat

        n_user = len(lbe_user.classes_)
        n_item = len(lbe_item.classes_)

        item_info = df_video_env.loc[lbe_item.classes_]
        item_info["item_id"] = item_info.index
        item_info = item_info[["item_id", "feat0", "feat1", "feat2", "feat3", "duration_normed"]]
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

    


# For loading KuaishouRec Data
@njit
def find_negative(user_ids, item_ids, mat_small, mat_big, df_negative, max_item):
    for i in range(len(user_ids)):
        user, item = user_ids[i], item_ids[i]

        neg = item + 1
        while neg <= max_item:
            if mat_small[user, neg] or mat_big[user, neg]:
                neg += 1
            else:
                df_negative[i, 0] = user
                df_negative[i, 1] = neg
                break
        else:
            neg = item - 1
            while neg >= 0:
                # if neg == 1225:  # 1225 is an absent item_id
                #     neg = 1224
                if mat_small[user, neg] or mat_big[user, neg]:
                    neg -= 1
                else:
                    df_negative[i, 0] = user
                    df_negative[i, 1] = neg
                    break


# For loading KuaiRec Data
def negative_sampling(df_train, df_item, df_user, y_name):
    small_path = os.path.join(DATAPATH, "small_matrix_processed.csv")
    df_small = pd.read_csv(small_path, header=0, usecols=['user_id', 'item_id'])

    mat_small = csr_matrix((np.ones(len(df_small)), (df_small['user_id'], df_small['item_id'])),
                           shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1),
                           dtype=np.bool).toarray()
    # df_negative = df_train.copy()
    mat_big = csr_matrix((np.ones(len(df_train)), (df_train['user_id'], df_train['item_id'])),
                         shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1), dtype=np.bool).toarray()

    # mat_negative = lil_matrix((df_train['user_id'].max() + 1, df_train['item_id'].max() + 1), dtype=np.bool).toarray()
    # find_negative(df_train['user_id'].to_numpy(), df_train['item_id'].to_numpy(), mat_small, mat_big, mat_negative,
    #               df_train['item_id'].max())
    # negative_pairs = np.array(list(zip(*mat_negative.nonzero())))
    # df_negative = pd.DataFrame(negative_pairs, columns=["user_id", "item_id"])
    # df_negative = df_negative[df_negative['item_id'] != 1225]  # 1225 is an absent item_id

    df_negative = np.zeros([len(df_train), 2])
    find_negative(df_train['user_id'].to_numpy(), df_train['item_id'].to_numpy(), mat_small, mat_big, df_negative,
                  df_train['item_id'].max())

    df_negative = pd.DataFrame(df_negative, columns=["user_id", "item_id"], dtype=int)
    df_negative = df_negative.merge(df_item, on=['item_id'], how='left')

    # video_mean_duration = KuaiEnv.load_video_duration()

    # df_negative['duration_normed'] = df_negative['item_id'].map(lambda x: video_mean_duration[x])
    # df_negative = df_negative.merge(video_mean_duration, on=['item_id'], how='left')

    df_negative[y_name] = 0.0

    return df_train, df_negative


def get_similarity_mat(list_feat, DATAPATH):
    similarity_mat_path = os.path.join(DATAPATH, "similarity_mat_video.csv")
    if os.path.isfile(similarity_mat_path):
        # with open(similarity_mat_path, 'rb') as f:
        #     similarity_mat = np.load(f, allow_pickle=True, fix_imports=True)
        print(f"loading similarity matrix... from {similarity_mat_path}")
        df_sim = pd.read_csv(similarity_mat_path, index_col=0)
        df_sim.columns = df_sim.columns.astype(int)
        print("loading completed.")
        similarity_mat = df_sim.to_numpy()
    else:
        series_feat_list = pd.Series(list_feat)
        df_feat_list = series_feat_list.to_frame("categories")
        df_feat_list.index.name = "video_id"

        similarity_mat = np.zeros([len(df_feat_list), len(df_feat_list)])
        print("Compute the similarity matrix (for the first time and will be saved for later usage)")
        for i in tqdm(range(len(df_feat_list)), desc="Computing..."):
            for j in range(i):
                similarity_mat[i, j] = similarity_mat[j, i]
            for j in range(i, len(df_feat_list)):
                similarity_mat[i, j] = len(set(series_feat_list[i]).intersection(set(series_feat_list[j]))) / len(
                    set(series_feat_list[i]).union(set(series_feat_list[j])))

        df_sim = pd.DataFrame(similarity_mat)
        df_sim.to_csv(similarity_mat_path)

    return similarity_mat


@njit
def compute_exposure_each_user(start_index: int,
                               distance_mat: np.ndarray,
                               timestamp: np.ndarray,
                               exposure_all: np.ndarray,
                               index_u: np.ndarray,
                               video_u: np.ndarray,
                               tau: float
                               ):
    for i in range(1, len(index_u)):
        video = video_u[i]
        t_diff = timestamp[index_u[i]] - timestamp[start_index:index_u[i]]
        t_diff[t_diff == 0] = 1  # important!
        # dist_hist = np.fromiter(map(lambda x: distance_mat[x, video], video_u[:i]), np.float)

        dist_hist = np.zeros(i)
        for j in range(i):
            video_j = video_u[j]
            dist_hist[j] = distance_mat[video_j, video]

        exposure = np.sum(np.exp(- t_diff * dist_hist / tau))
        exposure_all[start_index + i] = exposure


def compute_exposure_effect_kuaiRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH):
    exposure_path = os.path.join(MODEL_SAVE_PATH, "..", "saved_exposure", "exposure_pos_{:.1f}.csv".format(tau))

    if os.path.isfile(exposure_path):
        print("loading saved exposure scores: ", exposure_path)
        exposure_pos_df = pd.read_csv(exposure_path)
        exposure_pos = exposure_pos_df.to_numpy()
        return exposure_pos

    similarity_mat = get_similarity_mat(list_feat, DATAPATH)

    distance_mat = 1 / similarity_mat

    exposure_pos = np.zeros([len(df_x), 1])

    user_list = df_x["user_id"].unique()

    timestamp = timestamp.to_numpy()

    print("Compute the exposure effect (for the first time and will be saved for later usage)")
    for user in tqdm(user_list, desc="Computing exposure effect of historical data"):
        df_user = df_x[df_x['user_id'] == user]
        start_index = df_user.index[0]
        index_u = df_user.index.to_numpy()
        video_u = df_user['video_id'].to_numpy()
        compute_exposure_each_user(start_index, distance_mat, timestamp, exposure_pos,
                                   index_u, video_u, tau)

    exposure_pos_df = pd.DataFrame(exposure_pos)

    if not os.path.exists(os.path.dirname(exposure_path)):
        os.mkdir(os.path.dirname(exposure_path))
    exposure_pos_df.to_csv(exposure_path, index=False)

    return exposure_pos


def get_distance_mat(list_feat, sub_index_list, DATAPATH):
    if sub_index_list is not None:
        distance_mat_small_path = os.path.join(DATAPATH, "distance_mat_video_small.csv")
        if os.path.isfile(distance_mat_small_path):
            print("loading small distance matrix...")
            df_dist_small = pd.read_csv(distance_mat_small_path, index_col=0)
            df_dist_small.columns = df_dist_small.columns.astype(int)
            print("loading completed.")
        else:
            similarity_mat = get_similarity_mat(list_feat, DATAPATH)
            df_sim = pd.DataFrame(similarity_mat)
            df_sim_small = df_sim.loc[sub_index_list, sub_index_list]

            df_dist_small = 1.0 / df_sim_small

            df_dist_small.to_csv(distance_mat_small_path)

        return df_dist_small

    return None
