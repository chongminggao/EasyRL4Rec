import os
import pickle
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import LabelEncoder
from tqdm import tqdm

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = os.path.join(CODEPATH, "data")

REMOVE_THRESHOLD = 5

filename_GT = os.path.join(DATAPATH, "ground_truth_mat.pkl")
filename_df_train = os.path.join(DATAPATH, "df_train.csv")
filename_df_test = os.path.join(DATAPATH, "df_test.csv")
filename_df_item = os.path.join(DATAPATH, "df_item.csv")
filename_list_feat = os.path.join(DATAPATH, "list_feat.pkl")


cuda = 4
n_epochs = 20
batch_size = 4096
sgd_lr = 0.01
device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")




# 定义MF模型
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=8):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        # initialization
        self.user_factors.weight.data.normal_(0, 0.01)
        self.item_factors.weight.data.normal_(0, 0.01)

        
        hidden_dim = 8
        self.fc1 = nn.Linear(n_factors * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.fc1, self.relu, self.fc3, self.sigmoid)

    def forward(self, user, item):
        # logits = (self.user_factors(user) * self.item_factors(item)).sum(1)
        # res = self.relu(logits)

        logits = torch.concat((self.user_factors(user), self.item_factors(item)), dim=1)
        res = self.net(logits)

        return res


def load_category(lbe_item, lbe_item_processed):

    itemfeat_path = os.path.join(DATAPATH, "item-info.csv")
    df_item_feat = pd.read_csv(itemfeat_path, header=0, dtype={"itemId":"str", "category":"str"})
    df_item_feat.rename(columns={"itemId":"item_id"}, inplace=True)
    df_item_feat["item_id"] = lbe_item.transform(df_item_feat["item_id"])
    
    df_item_feat = df_item_feat[df_item_feat["item_id"].map(lambda x: x in lbe_item_processed.classes_)]
    df_item_feat["item_id"] = lbe_item_processed.transform(df_item_feat["item_id"])
    
    df_item_feat = df_item_feat.set_index("item_id")
    df_item_feat.sort_index(inplace=True)


    df_item_feat["category"] = df_item_feat["category"].map(lambda x: x.split("_and_") if type(x) == str else [])
    
    explored_feat = df_item_feat["category"].explode()

    lbe_item_feat = LabelEncoder()
    explored_feat = pd.DataFrame(lbe_item_feat.fit_transform(explored_feat), index=explored_feat.index, columns=["category"])
    explored_feat = explored_feat.reset_index()
    list_feat = explored_feat.groupby("item_id").agg(list)["category"].sort_index().to_list()

    max_num_feat = max(map(len, list_feat))

    df_feat = pd.DataFrame(list_feat, columns=[f'feat{i}' for i in range(max_num_feat)])
    df_feat.index.name = "item_id"
    df_feat[df_feat.isna()] = -1
    df_feat = df_feat + 1
    df_feat = df_feat.astype(int)

    return list_feat, df_feat
    


def load_and_preprocess():
    rawdata_path = os.path.join(DATAPATH, "reviews-info.csv")
    df_data = pd.read_csv(rawdata_path)
    df_data.rename(columns={"userId":"user_id"}, inplace=True)
    df_data.rename(columns={"itemId":"item_id"}, inplace=True)

    # df_data.loc[df_data["rating"] > 1, "rating"] = 1
    
    lbe_user = LabelEncoder()
    df_data["user_id"] = lbe_user.fit_transform(df_data["user_id"])

    lbe_item = LabelEncoder()
    df_data["item_id"] = lbe_item.fit_transform(df_data["item_id"])

    num_user = len(lbe_user.classes_)
    num_item = len(lbe_item.classes_)

    df_data["rating"].describe()
    
    # count df["rating"] with different values
    df_data["rating"].value_counts()
    # sum(df["rating"] > 0)

    df_data = df_data.sort_values(by=['user_id', 'timeStamp'])
    df_data.reset_index(drop=True, inplace=True)
    df_data_ori = df_data

    # Remove users and items that have less than 5 interactions
    remove_item_id = df_data.groupby("item_id")["user_id"].count() < REMOVE_THRESHOLD
    remove_ids = df_data["item_id"].map(lambda x: remove_item_id[x])
    df_data = df_data[~remove_ids]

    remove_user_id = df_data.groupby("user_id")["item_id"].count() < REMOVE_THRESHOLD
    remove_ids = df_data["user_id"].map(lambda x: remove_user_id[x])
    df_data = df_data[~remove_ids]
    
    print(f"reserved items: {df_data['item_id'].nunique()}")
    print(f"reserved users: {df_data['user_id'].nunique()}")
    
    df_data.reset_index(drop=True, inplace=True)
    lbe_item_processed = LabelEncoder()
    df_data["item_id"] = lbe_item_processed.fit_transform(df_data["item_id"])

    lbe_user_processed = LabelEncoder()
    df_data["user_id"] = lbe_user_processed.fit_transform(df_data["user_id"])

    list_feat, df_feat = load_category(lbe_item, lbe_item_processed)

    return df_data, list_feat, df_feat
    

class DatasetRS(torch.utils.data.Dataset):
    def __init__(self, df_data):
        self.df_data = df_data

    def __len__(self):
        return len(self.df_data)

    def compile(self):
        self.user_numpy = self.df_data["user_id"].to_numpy()
        self.item_numpy = self.df_data["item_id"].to_numpy()
        self.rating_numpy = self.df_data["rating"].to_numpy("float32")
        

    def __getitem__(self, index):
        user_id = self.user_numpy[index]
        item_id = self.item_numpy[index]
        rating = self.rating_numpy[index]
        
        return user_id, item_id, rating



def test_model(model, criterion, users_test, items_test, ratings_test):
    # 评估模型（你可能需要拆分数据集为训练集和验证集进行评估）
    model.eval()
    with torch.no_grad():    
        outputs = model(users_test, items_test)
        loss = criterion(outputs, ratings_test)
    # print(f"Test Loss: {loss.item():.4f}")
    return loss


def train_MF_model(df_data, df_train, df_test):

    # We use the df_test dataset to evaluate the model performance. Here we use MF to fill the missing values in the test dataset. So we use df_test to train the MF model and use df_train to evaluate the MF model.
    df_train_MF = df_test
    df_test_MF = df_train

    # df_train_MF = df_train
    # df_test_MF = df_test

    users_test = torch.LongTensor(df_test_MF['user_id'].values).to(device)
    items_test = torch.LongTensor(df_test_MF['item_id'].values).to(device)
    ratings_test = torch.FloatTensor(df_test_MF['rating'].values).to(device)

    # Use all the positive samples
    idx_test = ratings_test > 0

    # sample 3000 negative samples
    rand_index = np.random.choice(np.arange(len(df_test_MF)), size=3000, replace=False)
    idx_test[rand_index] = True

    users_test = users_test[idx_test]
    items_test = items_test[idx_test]
    ratings_test = ratings_test[idx_test]

    

    num_users = df_data["user_id"].nunique()
    num_items = df_data["item_id"].nunique()

    train_dataset = DatasetRS(df_train_MF)
    train_dataset.compile()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # test_dataset = DatasetRS(df_test)
    # test_dataset.compile()
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100000, shuffle=False)


    # 定义模型、损失函数和优化器
    model = MatrixFactorization(num_users, num_items).to(device)
    
    # criterion = nn.MSELoss()

    # weight = torch.Tensor([1, 10]).to(device)
    criterion = nn.functional.binary_cross_entropy_with_logits

    criterion_test = nn.MSELoss()
    criterion_test = nn.functional.mse_loss

    optimizer = optim.SGD(model.parameters(), lr=sgd_lr)
    # optimizer = optim.Adam(model.parameters(), lr=sgd_lr)

    # 训练模型
    
    epoch = -1
    test_loss = test_model(model, criterion_test, users_test, items_test, ratings_test)
    print(f"Epoch {epoch}/{n_epochs}, Training Loss: {0}, Test Loss: {test_loss.item():.4f}")
    
    for epoch in range(n_epochs):
        model.train()

        for idx, (users, items, ratings) in enumerate(train_loader):
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            outputs = model(users, items)
            
            outputs = outputs.squeeze()
            
            weight = torch.ones_like(ratings)
            weight[ratings > 0] = 10
            loss = criterion(outputs, ratings, weight=weight)

            # loss = criterion_test(outputs, ratings)

            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0 or epoch < 30:
            test_loss = test_model(model, criterion_test, users_test, items_test, ratings_test)
            print(f"Epoch {epoch}/{n_epochs}, Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    # 保存模型
    # users_all = torch.LongTensor(np.arange(num_users)).to(device)
    items_all = torch.LongTensor(np.arange(num_items)).to(device)

    all_res = []

    for user in tqdm(range(num_users), total=num_users, desc="Predicting..."):
        user_tensor = torch.LongTensor([user] * num_items).to(device)
        user_ratings = model(user_tensor, items_all).squeeze()
        
        all_res.append(user_ratings.cpu().detach().numpy())    
    mat = np.vstack(all_res)

    return mat

def save_data(df_train, df_test, list_feat, df_feat, mat):
    print("Save df_train to {}".format(filename_df_train))
    print("Save df_test to {}".format(filename_df_test))
    print("Save df_test to {}".format(filename_df_item))
    print("Save list_feat to {}".format(filename_list_feat))
    print("Save mat to {}".format(filename_GT))
    print("Saving data...")
    df_train.to_csv(filename_df_train, index=False)
    df_test.to_csv(filename_df_test, index=False)
    df_feat.to_csv(filename_df_item, index=True)
    pickle.dump(list_feat, open(filename_list_feat, "wb"))
    pickle.dump(mat, open(filename_GT, "wb"))


def split_data(df_data, split_ratio=0.5):
    
    df_data = df_data.sort_values(by=['user_id', 'timeStamp'])

    # 对每个用户进行切分
    train_data = []
    test_data = []

    for user_id, group in df_data.groupby('user_id'):
        n_items_for_user = len(group)
        n_train = int(split_ratio * n_items_for_user)
        
        train_data.append(group.iloc[:n_train])
        test_data.append(group.iloc[n_train:])

    df_train = pd.concat(train_data)
    df_test = pd.concat(test_data)

    # hard coded:
    line = df_test[df_test["item_id"]  == df_test["item_id"].max()].iloc[0:1]
    df_train = pd.concat([df_train, line])
    df_train.sort_values(by=['user_id', 'timeStamp'], inplace=True)
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_test



def create_ground_truth_mat():
    df_data, list_feat, df_feat = load_and_preprocess()
    df_train, df_test = split_data(df_data, split_ratio=0.5)

    
    mat = train_MF_model(df_data, df_train, df_test)

    print('df_data["user_id"].nunique():', df_data["user_id"].nunique())
    print('df_data["item_id"].nunique():', df_data["item_id"].nunique())
    print('df_train["user_id"].nunique():', df_train["user_id"].nunique())
    print('df_train["item_id"].nunique():', df_train["item_id"].nunique())
    print('df_test["user_id"].nunique():', df_test["user_id"].nunique())
    print('df_test["item_id"].nunique():', df_test["item_id"].nunique())
    print('df_train["user_id"].max():', df_train["user_id"].max())
    print('df_train["item_id"].max():', df_train["item_id"].max())
    print('df_test["user_id"].max():', df_test["user_id"].max())
    print('df_test["item_id"].max():', df_test["item_id"].max())
    print("mat.shape:", mat.shape)

    save_data(df_train, df_test, list_feat, df_feat, mat)
    print("Finished.")

    



if __name__ == "__main__":
    create_ground_truth_mat()
    
