# -*- coding: utf-8 -*-
# @Time    : 2021/9/11 4:24 下午
# @Author  : Chongming GAO
# @FileName: util.py
import os
import pickle
from logzero import logger

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix
from tqdm import tqdm

from core.util.inputs import SparseFeatP
from deepctr_torch.inputs import DenseFeat, VarLenSparseFeat


def create_dir(create_dirs):
    """
    create necessary dirs.
    """
    for dir in create_dirs:
        if not os.path.exists(dir):
            logger.info('Create dir: %s' % dir)
            try:
                os.mkdir(dir)
            except FileExistsError:
                print("The dir [{}] already existed".format(dir))


def compute_action_distance(action: np.ndarray, actions_hist: np.ndarray,
                            env_name="VirtualTB-v0", realenv=None):  # for kuaishou data
    if env_name == "VirtualTB-v0":
        a = action - actions_hist
        if len(a.shape) > 1:
            dist = np.linalg.norm(a, axis=1)
        else:
            dist = np.linalg.norm(a)
    elif env_name == "KuaiEnv-v0":
        # df_video_env = realenv.df_video_env
        # list_feat = realenv.list_feat
        # item_index = realenv.lbe_video.inverse_transform([action])
        # item_index_hist = realenv.lbe_video.inverse_transform(actions_hist)
        df_dist_small = realenv.df_dist_small
        dist = df_dist_small.iloc[action, actions_hist].to_numpy()
    else: # Coat
        dist = realenv.mat_distance[action, actions_hist]

    return dist


def compute_exposure(t_diff: np.ndarray, dist: np.ndarray, tau):
    if tau <= 0:
        res = 0
        return res
    res = np.sum(np.exp(- t_diff * dist / tau))
    return res


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def clip0(x):
    return np.amax(x, 0)




@njit
def find_negative(user_ids, item_ids, neg_u_list, neg_i_list, mat_train, df_negative, is_rand=True, num_break=3):
    """
    :param is_rand: Is sampling strategy a deterministic strategy.
    :type is_rand: bool
    """
    if is_rand:
        ind = 0
        for i in range(len(user_ids)):
            num_try = 0
            user, item = user_ids[i], item_ids[i]
            value = mat_train[user, item]
            while True:
                num_try += 1
                neg_u = neg_u_list[ind]
                neg_i = neg_i_list[ind]
                # neg_u = np.random.randint(max(user_ids) + 1)
                # neg_i = np.random.randint(max(item_ids) + 1)
                neg_v = mat_train[neg_u, neg_i]
                # if neg_v <= 0:

                ind = (ind + 1) % len(user_ids)
                if neg_v < value or num_try >= num_break:
                    break
            df_negative[i, 0] = neg_u
            df_negative[i, 1] = neg_i
            df_negative[i, 2] = neg_v
    else:
        for i in range(len(user_ids)):
            user, item = user_ids[i], item_ids[i]
            value = mat_train[user, item]

            neg_i = item + 1
            while neg_i < mat_train.shape[1]:
                neg_v = mat_train[user, neg_i]
                # if neg_v <= 0:
                if neg_v < value:
                    break
                neg_i += 1

            else:
                neg_i = item - 1
                while neg_i >= 0:
                    neg_v = mat_train[user, neg_i]
                    # if neg_v <= 0:
                    if neg_v < value:
                        break
                    neg_i -= 1

            df_negative[i, 0] = user
            df_negative[i, 1] = neg_i
            df_negative[i, 2] = neg_v



def align_ab(df_a, df_b):
    """
    len(df_b) > len(df_a)!!!
    """
    df_a.reset_index(drop=True,inplace=True)
    df_b.reset_index(drop=True, inplace=True)

    num_repeat = len(df_b) // len(df_a)
    df_ak = pd.concat([df_a] * int(num_repeat), ignore_index=True)
    num_rand = len(df_b) - len(df_ak)
    added_index = np.random.randint(low=0, high=len(df_a), size=num_rand)
    df_added = df_a.loc[added_index]

    df_a_res = pd.concat([df_ak, df_added], ignore_index=True)
    return df_a_res, df_b


def align_pos_neg(df_positive, df_negative, can_divide:bool):
    if can_divide:
        neg_K = len(df_negative) / len(df_positive)
        assert neg_K % 1 == 0
        df_pos = pd.concat([df_positive]*int(neg_K), ignore_index=True)
        df_neg = df_negative.reset_index(drop=True)
    else:
        if len(df_negative) > len(df_positive):
            df_pos, df_neg = align_ab(df_positive, df_negative)
        else:
            df_neg, df_pos = align_ab(df_negative, df_positive)

    return df_pos, df_neg

def negative_sampling(df_train, df_item, df_user, y_name, is_rand=True, neg_in_train=False, neg_K=5, num_break=3, sample_neg_popularity=False):
    print("negative sampling...")
    if neg_in_train: # 仅从已知样本中采负样本。
        neg_index = df_train[y_name] == 0
        pos_index = ~neg_index
        df_negative = df_train.loc[neg_index]
        df_positive = df_train.loc[pos_index]

        df_neg_K = pd.concat([df_negative] * neg_K, ignore_index=True)
        df_neg_K_permutated = df_neg_K.loc[np.random.permutation(len(df_neg_K))]

        df_pos, df_neg = align_pos_neg(df_positive, df_neg_K_permutated, can_divide=False)
    else:
        df_positive = df_train

        mat_train = csr_matrix((df_train[y_name], (df_train["user_id"], df_train["item_id"])),
                               shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1)).toarray()
        user_ids = df_train["user_id"].to_numpy()
        item_ids = df_train["item_id"].to_numpy()

        df_negative = pd.DataFrame([], columns=["user_id", "item_id", y_name])
        for k in tqdm(range(neg_K), desc="Negative sampling..."):
            array_k = np.zeros([len(df_train), 3])
            # neg_u_list = np.random.choice(min(user_ids), max(user_ids) + 1, size=len(user_ids) * num_break)
            # neg_i_list = np.random.choice(min(item_ids), max(item_ids) + 1, size=len(user_ids) * num_break)
            if sample_neg_popularity:
                neg_u_list = np.random.choice(user_ids, size=len(user_ids) * num_break)
                neg_i_list = np.random.choice(item_ids, size=len(user_ids) * num_break)
            else:
                neg_u_list = np.random.choice(list(set(user_ids)), size=len(user_ids) * num_break)
                neg_i_list = np.random.choice(list(set(item_ids)), size=len(user_ids) * num_break)
            find_negative(user_ids, item_ids, neg_u_list, neg_i_list, mat_train, array_k, is_rand=is_rand, num_break=num_break)
            df_k = pd.DataFrame(array_k, columns=["user_id", "item_id", y_name])
            # df_negative = df_negative.append(df_k, ignore_index=True)
            df_negative = pd.concat([df_negative, df_k])

        df_negative = df_negative.join(df_item, on=['item_id'], how="left")
        df_negative = df_negative.join(df_user, on=['user_id'], how="left")

        df_pos, df_neg = align_pos_neg(df_positive, df_negative, can_divide=True)

    return df_pos, df_neg


def compute_input_dim(feature_columns, include_sparse=True, include_dense=True, feature_group=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, (SparseFeatP, VarLenSparseFeat)), feature_columns)) if len(
        feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    dense_input_dim = sum(
        map(lambda x: x.dimension, dense_feature_columns))
    if feature_group:
        sparse_input_dim = len(sparse_feature_columns)
    else:
        sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
    input_dim = 0
    if include_sparse:
        input_dim += sparse_input_dim
    if include_dense:
        input_dim += dense_input_dim
    return input_dim