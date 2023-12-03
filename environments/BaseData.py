import pickle
from abc import ABC
import numpy as np
from numba import njit
import collections
from logzero import logger
import os

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