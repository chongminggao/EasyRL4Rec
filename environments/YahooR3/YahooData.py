import os
import sys
import pickle
import pandas as pd
import numpy as np

from environments.BaseData import BaseData, get_distance_mat
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

ROOTPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")


class YahooData(BaseData):
    def __init__(self):
        super(YahooData, self).__init__()
        self.train_data_path = "ydata-ymusic-rating-study-v1_0-train.txt"
        self.val_data_path = "ydata-ymusic-rating-study-v1_0-test.txt"
        
    def get_features(self, is_userinfo=None):
        user_features = ["user_id"]
        item_features = ['item_id']
        reward_features = ["rating"]
        return user_features, item_features, reward_features

    def get_df(self, name="ydata-ymusic-rating-study-v1_0-train.txt"):
        # read interaction
        filename = os.path.join(DATAPATH, name)
        df_data = pd.read_csv(filename, sep="\s+", header=None, names=["user_id", "item_id", "rating"])

        df_data["user_id"] -= 1
        df_data["item_id"] -= 1

        df_user = self.load_user_feat()
        df_item = self.load_item_feat()
        list_feat = None

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        return None
    
    def get_item_similarity(self):
        item_similarity_path = os.path.join(PRODATAPATH, "item_similarity.pickle")
        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            mat = YahooData.load_mat()
            mat_distance = YahooData.get_saved_distance_mat(mat)
            item_similarity = 1 / (mat_distance + 1)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
      
    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            df_data, df_user, df_item, list_feat = self.get_df("ydata-ymusic-rating-study-v1_0-train.txt")

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

    def load_user_feat(self):
        df_user = pd.DataFrame(np.arange(15400), columns=["user_id"])
        df_user.set_index("user_id", inplace=True)
        return df_user

    def load_item_feat(self):
        df_item = pd.DataFrame(np.arange(1000), columns=["item_id"])
        df_item.set_index("item_id", inplace=True)
        return df_item


    @staticmethod
    def load_mat():
        filename_GT = os.path.join(DATAPATH, "RL4Rec_data", "yahoo_pseudoGT_ratingM.ascii")
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)
        return mat

    @staticmethod
    def get_saved_distance_mat(mat):
        distance_mat_path = os.path.join(PRODATAPATH, f"distance_mat.pickle")
        if os.path.isfile(distance_mat_path):
            mat_distance = pickle.load(open(distance_mat_path, "rb"))
        else:
            num_item = mat.shape[1]
            distance = np.zeros([num_item, num_item])
            mat_distance = get_distance_mat(mat, distance)
            pickle.dump(mat_distance, open(distance_mat_path, 'wb'))
        return mat_distance
    