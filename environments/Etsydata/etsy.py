import os
import pickle

import gymnasium as gym
from sklearn.calibration import LabelEncoder
import torch
from gym import spaces
from numba import njit


import pandas as pd
import numpy as np
import random

from tqdm import tqdm
import sys

from environments.BaseEnv import BaseEnv
from environments.Etsydata.process_data import create_ground_truth_mat
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.util.utils import get_sorted_domination_features

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = os.path.join(CODEPATH, "data")

NUM_USER = 1283 # hard coded!

class EtsyEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, df_item=None, mat_distance=None, num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):

        if mat is not None:
            self.mat = mat
            self.df_item = df_item
            self.mat_distance = mat_distance
        else:
            self.mat, self.df_item, self.mat_distance = self.load_mat()

        super(EtsyEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def get_df_etsy(name):
        # read interaction
        filename = os.path.join(DATAPATH, name)
        df_data = pd.read_csv(filename, header=0)

        # read user feature
        df_user = EtsyEnv.load_user_feat()

        # read item features
        list_feat, df_item = EtsyEnv.load_category()


        df_data = df_data.join(df_user, on="user_id", how='left')
        df_data = df_data.join(df_item, on="item_id", how='left')


        return df_data, df_user, df_item, list_feat

    @staticmethod
    def load_category():
        # filename_df_item = os.path.join(DATAPATH, "df_item.csv")
        # if os.path.isfile(filename_df_item) and os.path.isfile(filename_list_feat):
        filename_list_feat = os.path.join(DATAPATH, "list_feat.pkl")
        list_feat = pickle.load(open(filename_list_feat, 'rb'))

        df_item = EtsyEnv.load_item_feat()

        return list_feat, df_item

    @staticmethod
    def get_domination():
        df_data, _, df_item, _ = EtsyEnv.get_df_etsy("df_train.csv")
        CODEDIRPATH = os.path.dirname(__file__)
        feature_domination_path = os.path.join(CODEDIRPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = get_sorted_domination_features(
                df_data, df_item, yname="rating", is_multi_hot=True, threshold=1.0)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination

    @staticmethod
    def load_exposure_and_popularity(predicted_mat, filename="df_train.csv"):
        filepath = os.path.join(DATAPATH, filename)
        mat_train = pd.read_csv(filepath, sep="\s+", header=None)
        isexposure = mat_train > 0
        df_popularity = isexposure.sum()

        df_train, _, _ = EtsyEnv.get_df_etsy("df_train.csv")
        df_frequency = df_train.groupby(["user_id", "item_id"])["rating"].agg(len)

        return df_frequency, df_popularity

    @staticmethod
    def get_item_popularity():
        CODEDIRPATH = os.path.dirname(__file__)
        item_popularity_path = os.path.join(CODEDIRPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            df_data, df_user, df_item, list_feat = EtsyEnv.get_df_etsy("df_train.csv")

            n_users = df_data['user_id'].nunique()
            n_items = df_data['item_id'].nunique()

            df_data_filtered = df_data[df_data["rating"]>=3.]
            
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
    def get_item_similarity():

        mat, df_item, mat_distance = EtsyEnv.load_mat()
        item_similarity = DynamicArray(lambda x, y: 1 / (mat_distance[x, y] + 1))
        
        # CODEDIRPATH = os.path.dirname(__file__)
        # item_similarity_path = os.path.join(CODEDIRPATH, "item_similarity.pickle")

        # if os.path.isfile(item_similarity_path):
        #     item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        # else:
        #     mat, df_item, mat_distance = EtsyEnv.load_mat()
        #     item_similarity = 1 / (mat_distance + 1)
        #     pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity

        

    @staticmethod
    def load_mat():
        filename_GT = os.path.join(DATAPATH, "ground_truth_mat.pkl")
        if os.path.exists(filename_GT):
            mat = pickle.load(open(filename_GT, 'rb'))
        else:
            mat = create_ground_truth_mat()
            # mat.to_csv(filename_GT, header=0, index=False)


        # mat_distance = get_distance_mat(mat)

        num_item = mat.shape[1]
        
        # distance = np.zeros([num_item, num_item])
        # mat_distance = get_distance_mat1(mat, distance)

        def func(item1, item2):
            matt = np.transpose(mat)
            vec_i = matt[item1]
            vec_j = matt[item2]
            dist = ((vec_i - vec_j) ** 2).sum() ** 0.5
            return dist

        mat_distance = DynamicArray(func)

        df_item = EtsyEnv.load_item_feat()

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
        df_user = pd.DataFrame(np.arange(NUM_USER), columns=["user_id"])
        df_user.set_index("user_id", inplace=True)
        return df_user

    @staticmethod
    def load_item_feat():
        filename_df_item = os.path.join(DATAPATH, "df_item.csv")
        df_item = pd.read_csv(filename_df_item, header=0)
        df_item.set_index("item_id", inplace=True)

        return df_item

    


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

    
class DynamicArray:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 2:
            i, j = idx
            return self.func(i, j)
        else:
            raise ValueError("Expected indices i, j in the form mat[i, j]")

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
    df_item = EtsyEnv.load_item_feat()
    df_user = EtsyEnv.load_user_feat()

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
