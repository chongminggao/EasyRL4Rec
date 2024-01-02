import pickle
from abc import ABC
import numpy as np
import pandas as pd
from numba import njit
import collections
from logzero import logger
import os

from tqdm import tqdm


class BaseData(ABC):

    def __init__(self):
        self.train_data_path = None
        self.val_data_path = None

    def get_features(self, is_userinfo=True):
        pass

    def get_df(self, name=None):
        pass

    def get_train_data(self):
        return self.get_df(self.train_data_path)

    def get_val_data(self):
        return self.get_df(self.val_data_path)

    def get_domination(self):
        pass
    
    def get_item_similarity(self):
        pass
        
    def get_item_popularity(self):
        pass

    def get_sorted_domination_features(self, df_data, df_item, is_multi_hot, yname=None, threshold=None):
        """
        :param threshold: is used for counting only the positive samples.
        """
        item_feat_domination = dict()
        if not is_multi_hot: # one-hot for coat
            item_feat = df_item.columns.to_list()
            for x in item_feat:
                sorted_count = collections.Counter(df_data[x])
                sorted_percentile = dict(map(lambda x: (x[0], x[1] / len(df_data)), dict(sorted_count).items()))
                sorted_items = sorted(sorted_percentile.items(), key=lambda x: x[1], reverse=True)
                item_feat_domination[x] = sorted_items
        else: # multi-hot for kuairec and kuairand
            df_item_filtered = df_item.filter(regex="^feat", axis=1)

            # df_item_flat = df_item_filtered.to_numpy().reshape(-1)
            # df_item_nonzero = df_item_flat[df_item_flat>0]

            feat_train = df_data.loc[df_data[yname] >= threshold, df_item_filtered.columns.to_list()]
            cats_train = feat_train.to_numpy().reshape(-1)
            pos_cat_train = cats_train[cats_train > 0]

            sorted_count = collections.Counter(pos_cat_train)
            sorted_percentile = dict(map(lambda x: (x[0], x[1] / sum(sorted_count.values())), dict(sorted_count).items()))
            sorted_items = sorted(sorted_percentile.items(), key=lambda x: x[1], reverse=True)

            item_feat_domination["feat"] = sorted_items

        return item_feat_domination
    
    @staticmethod
    def get_saved_distance_mat(mat, PRODATAPATH):
        distance_mat_path = os.path.join(PRODATAPATH, f"distance_mat.pickle")
        if os.path.isfile(distance_mat_path):
            mat_distance = pickle.load(open(distance_mat_path, "rb"))
        else:
            num_item = mat.shape[1]
            distance = np.zeros([num_item, num_item])
            logger.info("Compute the distance matrix for the first time. It may take some time... \n(The computed data will be saved for the further usage).")
            mat_distance = get_distance_mat(mat, distance)
            pickle.dump(mat_distance, open(distance_mat_path, 'wb'))
            logger.info(f"The computed distance matrix has been saved in {distance_mat_path}")
        return mat_distance

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


@njit
def compute_exposure_each_user(distance_mat: np.ndarray,
                               timestamp: np.ndarray,
                               exposure_all: np.ndarray,
                               index_u: np.ndarray,
                               video_u: np.ndarray,
                               tau: float,
                               window:int = 30,
                               ):

    for i in range(1, len(index_u)):
        start_index = max(0, i - window)
        t_diff = timestamp[index_u[i]] - timestamp[index_u[start_index:i]]
        t_diff[t_diff == 0] = 1  # important!
        # dist_hist = np.fromiter(map(lambda x: distance_mat[x, video], video_u[:i]), np.float)

        dist_hist = np.zeros(min(window, i))
        for ind, j in enumerate(range(start_index, i)):
            dist_hist[ind] = distance_mat[video_u[j], video_u[i]]

        exposure = np.sum(np.exp(- t_diff * dist_hist / tau))
        exposure_all[index_u[i]] = exposure

def compute_exposure_effect(dataset, df_pos, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH, window=30):
    exposure_path = os.path.join(MODEL_SAVE_PATH, "..", "saved_exposure", "exposure_pos_{:.1f}.csv".format(tau))

    if os.path.isfile(exposure_path):
        print("loading saved exposure scores: ", exposure_path)
        exposure_pos_df = pd.read_csv(exposure_path)
        exposure_pos = exposure_pos_df.to_numpy()
        return exposure_pos

    similarity_mat = dataset.get_item_similarity()

    distance_mat = 1 / (similarity_mat + 0.001)

    exposure_pos = np.zeros([len(df_pos), 1])

    user_list = df_pos["user_id"].unique()

    timestamp = np.array(timestamp)

    print("Compute the exposure effect (for the first time and will be saved for later usage)")
    for user in tqdm(user_list, desc="Computing exposure effect of historical data"):
        df_user = df_pos[df_pos['user_id'] == user]
        index_u = df_user.index.to_numpy()
        video_u = df_user['item_id'].to_numpy()
        compute_exposure_each_user(distance_mat, timestamp, exposure_pos, index_u, video_u, tau, window=window)

    exposure_pos_df = pd.DataFrame(exposure_pos)

    if not os.path.exists(os.path.dirname(exposure_path)):
        os.mkdir(os.path.dirname(exposure_path))
    exposure_pos_df.to_csv(exposure_path, index=False)

    return exposure_pos