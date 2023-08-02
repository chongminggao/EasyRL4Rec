# -*- coding: utf-8 -*-
# @Time    : 2022/11/14 14:31
# @Author  : Chongming GAO
# @FileName: static_dataset.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class StaticDataset(Dataset):
    def __init__(self, x_columns, y_columns, num_workers=4):
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.num_workers = num_workers
        self.len = 0
        self.neg_items_info = None
        self.ground_truth = None

        self.all_item_ranking = False

    def set_all_item_ranking_in_evaluation(self, all_item_ranking):
        self.all_item_ranking = all_item_ranking

    def set_df_user_val(self, df_user_val):
        self.df_user_val = df_user_val
        self.df_user_val.sort_index(inplace=True)

    def set_df_item_val(self, df_item_val):  # for kuaishou data
        self.df_item_val = df_item_val
        self.df_item_val.sort_index(inplace=True)

    def set_ground_truth(self, ground_truth):  # for kuaishou data
        self.ground_truth = ground_truth

    def set_user_col(self, ind):
        self.user_col = ind

    def set_item_col(self, ind):
        self.item_col = ind

    def set_dataset_complete(self, dataset):
        self.dataset_complete = dataset

    def compile_dataset(self, df_x, df_y, score=None):
        self.x_numpy = df_x.to_numpy()
        self.y_numpy = df_y.to_numpy()

        if score is None:
            self.score = np.zeros([len(self.x_numpy), 1])
        else:
            self.score = score

        self.x_numpy = self.x_numpy.astype(float)
        self.y_numpy = self.y_numpy.astype(float)

        self.len = len(self.x_numpy)

    def get_dataset_train(self):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.x_numpy),
                                                 torch.from_numpy(self.y_numpy),
                                                 torch.from_numpy(self.score))
        return dataset

    def get_dataset_eval(self):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.x_numpy),
                                                 torch.from_numpy(self.y_numpy))
        return dataset

    def get_y(self):
        return self.y_numpy

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        x = self.x_numpy[index]
        y = self.y_numpy[index]
        return x, y

