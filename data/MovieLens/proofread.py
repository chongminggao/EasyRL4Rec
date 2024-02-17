import pandas as pd
import numpy as np
import os
import sklearn
from scipy.sparse import coo_matrix

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "data_raw")

df_mat  = pd.read_csv(os.path.join(DATAPATH,"rating_matrix.csv"), header=None)
mat = df_mat.to_numpy()
# mat = np.ones_like(mat) * 3.5

train_data = pd.read_csv(os.path.join(DATAPATH,"movielens-1m-train.csv"))
test_data = pd.read_csv(os.path.join(DATAPATH,"movielens-1m-test.csv"))


def get_dense_mat(df_data):
    num_user = 6040
    num_item = 3952
    sparse_matrix = coo_matrix((df_data["Rating"], (df_data["UserID"], df_data["MovieID"])), shape=(num_user + 1, num_item + 1))
    dense_matrix = sparse_matrix.toarray()
    return dense_matrix

train_mat = get_dense_mat(train_data)[1:,1:]
test_mat = get_dense_mat(test_data)[1:,1:]


def get_mae(my_mat, mat, mask):
    diff = my_mat - mat
    squared_diff = np.abs(diff) * mask
    mae = np.sum(squared_diff) / mask.sum()
    return mae


mask_train = train_mat > 0
mask_test = test_mat > 0

mae_train = get_mae(train_mat, mat, mask_train)
print(mae_train)

mae_test = get_mae(test_mat, mat, mask_test)
print(mae_test)

a = 1


