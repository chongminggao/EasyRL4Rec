import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from environments.MovieLens import provide_MF_results
from environments.BaseData import BaseData, get_distance_mat

ROOTPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")

for path in [PRODATAPATH]:
    if not os.path.exists(path):
        os.mkdir(path)


class MovieLensData(BaseData):
    def __init__(self):
        super(MovieLensData, self).__init__()
        self.train_data_path = "movielens-1m-train.csv"
        self.val_data_path = "movielens-1m-test.csv"

    def get_features(self, is_userinfo=True):
        user_features = ["user_id", "gender", "age_range", "occupation"]
        if not is_userinfo:
            user_features = ["user_id"]    
        item_features = ['item_id'] + ["feat{}".format(i) for i in range(6)]
        reward_features = ["rating"]
        return user_features, item_features, reward_features

    def get_df(self, name="movielens-1m-train.csv"):
        # read interaction
        filename = os.path.join(DATAPATH, name)
        # df_data = pd.read_csv(filename, sep="\s+", header=None, names=["user_id", "item_id", "rating"])
        df_data = pd.read_csv(filename, header=0, names=["user_id", "item_id", "rating", "timestamp"])

        # df_data["user_id"] -= 1
        # df_data["item_id"] -= 1
        df_user = MovieLensData.load_user_feat()
        list_feat, df_item = MovieLensData.load_category()

        df_data = df_data.join(df_user, on="user_id", how='left')
        df_data = df_data.join(df_item, on="item_id", how='left')

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        df_data, _, df_item, _ = self.get_df("movielens-1m-train.csv")
        feature_domination_path = os.path.join(PRODATAPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = self.get_sorted_domination_features(
                df_data, df_item, is_multi_hot=True, yname="rating", threshold=4)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination

    def get_item_similarity(self):
        item_similarity_path = os.path.join(PRODATAPATH, "item_similarity.pickle")
        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            mat = MovieLensData.load_mat()
            mat_distance = MovieLensData.get_saved_distance_mat(mat)
            item_similarity = 1 / (mat_distance + 1)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity

    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            df_data, df_user, df_item, list_feat = self.get_df("movielens-1m-train.csv")

            n_users = df_user.index.nunique()
            n_items = df_item.index.nunique()

            df_data_filtered = df_data[df_data["rating"] >= 4.]

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
    def load_category(tag_label="tags"):
        
        filepath = os.path.join(DATAPATH, 'movies.dat')
        df_item = pd.read_csv(filepath, 
                              sep="::", 
                              header=None,
                              names=["item_id", "movie_title", "genre"],
                              dtype={0: int, 1: str, 2: str},
                              encoding='latin1', engine='python')  # Specify the encoding as 'latin1'

        df_item["release_year"] = df_item["movie_title"].apply(lambda x: x[-5:-1])
        df_item["release_year"] = df_item["release_year"].astype(int)
        df_item["movie_title"] = df_item["movie_title"].apply(lambda x: x[:-7])

        df_item["genre"] = df_item["genre"].apply(lambda x: x.split("|"))
        df_item["num_genre"] = df_item["genre"].apply(lambda x: len(x))

        df_item.set_index("item_id", inplace=True)
        df_item_all = df_item.reindex(list(range(df_item.index.min(),df_item.index.max()+1)))
        df_item_all.loc[df_item_all["genre"].isna(), "genre"] = df_item_all.loc[df_item_all["genre"].isna(), "genre"].apply(lambda x: [])

        list_feat = df_item_all['genre'].to_list()

        # df_item["year_range"] = pd.cut(df_item["release_year"], bins=[1900, 1950, 2000, 2050], labels=False)

        df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3', 'feat4', 'feat5'], index=df_item_all.index)
        
        lbe = LabelEncoder()
        tags_array = lbe.fit_transform(df_feat.to_numpy().reshape(-1)).reshape(df_feat.shape)
        tags_array += 1
        tags_array[tags_array.max() == tags_array] = 0
        df_feat = pd.DataFrame(tags_array, columns=df_feat.columns, index=df_feat.index)

        list_feat_num = list(map(lambda x: lbe.transform(x) + 1, list_feat))
        df_feat[tag_label] = list_feat_num
        
        return list_feat_num, df_feat

    @staticmethod
    def load_item_feat():
        list_feat, df_feat = MovieLensData.load_category()
        df_item = df_feat
        return df_item

    @staticmethod
    def load_user_feat():
        # df_user = pd.DataFrame(np.arange(6040), columns=["user_id"])
        # df_user = pd.read_csv(os.path.join(DATAPATH, "users.dat"), sep="::", engine="python", header=None, names=["user_id", "gender",
                                                                                                                    #   def load_user_feat():
        print("load user features")
        filepath = os.path.join(DATAPATH, "users.dat")
        df_user = pd.read_csv(filepath, sep="::", header=None, names=["user_id", "gender", "age", "occupation", "zip_code"], dtype={0: int, 1: str, 2: int, 3: int, 4: str}, engine="python")
        df_user["zip_code"] = df_user["zip_code"].apply(lambda x: x.split("-")[0])
        
        
        age_range = [0, 18, 25, 35, 45, 50, 56]
        df_user['age_range'] = pd.cut(df_user['age'], bins=age_range, labels=False)
        df_user['age_range'] += 1

        for col in ['gender', 'occupation', 'zip_code']:
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            
            df_user[col] += 1

        df_user.set_index("user_id", inplace=True)
        
        return df_user


    # def load_item_feat(self):
    #     df_item = pd.DataFrame(np.arange(3952), columns=["item_id"])
    #     df_item.set_index("item_id", inplace=True)
    #     return df_item

    @staticmethod
    def load_mat():
        filename_GT = os.path.join(DATAPATH, "rating_matrix.csv")
        if os.path.exists(filename_GT):
            mat = pd.read_csv(filename_GT, header=None).to_numpy()
        else:
            mat = provide_MF_results.main()
            
        mat[mat < 0] = 0
        mat[mat > 5] = 5
        return mat

    @staticmethod
    def get_lbe():
        df_user = MovieLensData.load_user_feat()
        df_item = MovieLensData.load_item_feat()
        lbe_user = LabelEncoder()
        lbe_item = LabelEncoder()
        lbe_user.fit(df_user.index)
        lbe_item.fit(df_item.index)

        return lbe_user, lbe_item

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


if __name__ == "__main__":
    dataset = MovieLensData()
    df_train, df_user_train, df_item_train, _ = dataset.get_train_data()
    df_val, df_user_val, df_item_val, _ = dataset.get_val_data()
    print("MovieLens-1M: Train #user={}  #item={}  #inter={}".format(df_train['user_id'].nunique(),
                                                                     df_train['item_id'].nunique(), len(df_train)))
    print("MovieLens-1M: Test  #user={}  #item={}  #inter={}".format(df_val['user_id'].nunique(),
                                                                     df_val['item_id'].nunique(), len(df_val)))
