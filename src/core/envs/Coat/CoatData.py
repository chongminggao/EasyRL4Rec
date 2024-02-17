import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from src.core.envs.BaseData import BaseData, get_distance_mat

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/Coat"
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")

for path in [PRODATAPATH]:
    if not os.path.exists(path):
        os.mkdir(path)

class CoatData(BaseData):
    def __init__(self):
        super(CoatData, self).__init__()
        self.train_data_path = "train.ascii"
        self.val_data_path = "test.ascii"
    
    def get_features(self, is_userinfo=True):
        user_features = ["user_id", 'gender_u', 'age', 'location', 'fashioninterest']
        if not is_userinfo:
            user_features = ["user_id"]
        item_features = ['item_id', 'gender_i', "jackettype", 'color', 'onfrontpage']
        reward_features = ["rating"]
        return user_features, item_features, reward_features
        
    def get_df(self, name="train.ascii"):
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
        df_user = self.load_user_feat()

        # read item features
        df_item = CoatData.load_item_feat()

        df_data = df_data.join(df_user, on="user_id", how='left')
        df_data = df_data.join(df_item, on="item_id", how='left')

        df_data = df_data.astype(int)
        list_feat = None

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        df_data, _, df_item, _ = self.get_df("train.ascii")
        feature_domination_path = os.path.join(PRODATAPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = self.get_sorted_domination_features(
                df_data, df_item, is_multi_hot=False)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination
    
    def get_item_similarity(self):
        item_similarity_path = os.path.join(PRODATAPATH, "item_similarity.pickle")
        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            mat = CoatData.load_mat()
            mat_distance = CoatData.get_saved_distance_mat(mat, PRODATAPATH)
            item_similarity = 1 / (mat_distance + 1)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
      
    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            df_data, df_user, df_item, list_feat = self.get_df("train.ascii")

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
        filename_user = os.path.join(DATAPATH, "user_features.ascii")
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
        filename_item = os.path.join(DATAPATH, "item_features.ascii")
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
    
    @staticmethod
    def load_mat():
        # Note: The data file `coat_pseudoGT_ratingM.ascii` is sourced from the https://github.com/BetsyHJ/RL4Rec repository.
        filename_GT = os.path.join(DATAPATH, "RL4Rec_data", "coat_pseudoGT_ratingM.ascii")
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)
        return mat


 
if __name__ == "__main__":
    dataset = CoatData()
    df_train, df_user_train, df_item_train, _ = dataset.get_train_data()
    df_val, df_user_val, df_item_val, _ = dataset.get_val_data()
    print("Coat: Train #user={}  #item={}  #inter={}".format(df_train['user_id'].nunique(), df_train['item_id'].nunique(), len(df_train)))
    print("Coat: Test  #user={}  #item={}  #inter={}".format(df_val['user_id'].nunique(), df_val['item_id'].nunique(), len(df_val)))
