import os
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np

test_size=0.2
# Load the Movielens ratings data
CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "data_raw")
ratings_df = pd.read_csv(os.path.join(DATAPATH, "ratings.dat"), delimiter='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
df_item = pd.read_csv(os.path.join(DATAPATH, "movies.dat"), delimiter='::', header=None, names=['MovieID', 'Title', 'Genres'], engine='python')

ratings_df = ratings_df.sort_values(by=['UserID', 'Timestamp']).reset_index(drop=True)

# 以用户ID为分组依据
grouped = ratings_df.groupby('UserID')

train_list = []
test_list = []
# Function to ensure all items are in the training set
def ensure_train_coverage(train, test):
    test_items = set(test['MovieID'])
    train_items = set(train['MovieID'])

    # Items only in test set
    exclusive_test_items = test_items - train_items

    for item in exclusive_test_items:
        # Move one instance of this item to the training set
        item_idx = test[test['MovieID'] == item].index[0]
        train = pd.concat([train, test.loc[item_idx:item_idx]])
        test = test.drop(item_idx)

    return train, test

# Splitting data ensuring all items are in training set
for _, group in grouped:
    train, test = train_test_split(group, test_size=0.2, random_state=42)
    train, test = ensure_train_coverage(train, test)
    train_list.append(train)
    test_list.append(test)

# Combine into final training and test sets
train = pd.concat(train_list)
test = pd.concat(test_list)


ratings_df.nunique()
train.nunique()
train.nunique()





a = 1