# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 9:49 上午
# @Author  : Chongming GAO
# @FileName: state_tracker.py

import numpy as np
import torch
from torch import nn, Tensor

from core.util.inputs import input_from_feature_columns
from core.userModel.user_model import build_input_features
from core.util.utils import compute_input_dim
from deepctr_torch.inputs import combined_dnn_input

FLOAT = torch.FloatTensor


def reverse_padded_sequence(tensor: Tensor, lengths: Tensor):
    """
    Change the input tensor from:
    [[1, 2, 3, 4, 5],
    [1, 2, 0, 0, 0],
    [1, 2, 3, 0, 0]]
    to:
    [[5, 4, 3, 2, 1],
    [2, 1, 0, 0, 0],
    [3, 2, 1, 0, 0]]
    :param tensor: (B, max_length, *)
    :param lengths: (B,)
    :return:
    """
    out = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        out[i, :lengths[i]] = tensor[i, :lengths[i]].flip(dims=[0])
    return out


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


class StateTracker_Base(nn.Module):
    def __init__(self, user_columns, action_columns, feedback_columns, 
                 dim_model,
                 train_max=None, 
                 train_min=None, 
                 test_max=None, 
                 test_min=None, 
                 reward_handle=None,
                 saved_embedding=None,
                 device="cpu",
                 window_size=10, use_userEmbedding=False):
        super().__init__()
        self.user_columns = user_columns
        self.action_columns = action_columns
        self.feedback_columns = feedback_columns
        self.use_userEmbedding = use_userEmbedding

        self.user_index = build_input_features(user_columns)
        self.action_index = build_input_features(action_columns)
        self.feedback_index = build_input_features(feedback_columns)


        self.dim_model = dim_model
        self.window_size = window_size
        # self.random_init = random_init
        self.device = device

        self.test_min = test_min
        self.test_max = test_max
        self.train_min = train_min
        self.train_max = train_max
        self.reward_handle = reward_handle


        self.num_user = user_columns[0].vocabulary_size
        self.num_item = action_columns[0].vocabulary_size
        self.hidden_size = action_columns[0].embedding_dim
        
        if self.reward_handle == "cat" or self.reward_handle == "cat2":
            self.hidden_size += 1
        
        if saved_embedding is None:
            embedding_dict = torch.nn.ModuleDict({
                "feat_item": torch.nn.Embedding(
                    num_embeddings=self.num_item + 1, embedding_dim=self.hidden_size),
                "feat_user": torch.nn.Embedding(
                    num_embeddings=self.num_user, embedding_dim=self.hidden_size)
                })
            self.embedding_dict = embedding_dict.to(device)

            # Initialize embedding
            nn.init.normal_(self.embedding_dict.feat_item.weight, 0, 0.01)
        else:

            self.embedding_dict = saved_embedding.to(device)

            # self.num_item = self.embedding_dict.feat_item.weight.shape[0]
            # self.dim_item = self.embedding_dict.feat_item.weight.shape[1]

            # Add a new embedding vector for a non-exist item.
            new_embedding = FLOAT(1, self.dim_model).to(device)
            nn.init.normal_(new_embedding, mean=0, std=0.01)
            emb_cat = torch.cat([self.embedding_dict.feat_item.weight.data, new_embedding])
            new_item_embedding = torch.nn.Embedding.from_pretrained(
                emb_cat, freeze=not self.embedding_dict.feat_item.weight.requires_grad)
            self.embedding_dict.feat_item = new_item_embedding

        if self.use_userEmbedding:
            pass 
            # Todo
            # self.ffn_user = nn.Linear(compute_input_dim(self.user_columns), self.dim_model, device=self.device)
            
    
    def get_embedding(self, X, type):
        if type == "user":
            feat_columns = self.user_columns
            feat_index = self.user_index
        elif type == "action":
            feat_columns = self.action_columns
            feat_index = self.action_index
        elif type == "feedback":
            feat_columns = self.feedback_columns
            feat_index = self.feedback_index

        X[X == -1] = self.num_item

        sparse_embedding_list, dense_value_list = input_from_feature_columns(FLOAT(X).to(self.device), feat_columns,
                                                                             self.embedding_dict, feat_index,
                                                                             support_dense=True, device=self.device)
        new_X = combined_dnn_input(sparse_embedding_list, dense_value_list)
        X_res = new_X

        return X_res

    def get_normed_reward(self, e_r, is_train):
        if is_train:
            r_max = self.train_max
            r_min = self.train_min
        else:
            r_max = self.test_max
            r_min = self.test_min

        if r_max is not None and r_min is not None:
            normed_r = (e_r - r_min) / (r_max - r_min)
            # if not (all(normed_r<=1) and all(normed_r>=0)):
            #     a = 1
            normed_r[normed_r>1] = 1 # todo: corresponding to the initialize reward line above.
            # assert (all(normed_r<=1) and all(normed_r>=0))
        else:
            normed_r = e_r

        return normed_r

    def convert_to_k_state_embedding(self, buffer=None, indices=None, is_obs=None, batch=None, use_batch_in_statetracker=False, is_train=True):
        if use_batch_in_statetracker: # when collector collects the data, batch is not None.
            assert batch is not None
            user_item_pair_all = batch.obs
            rew_all = batch.rew_prev
            live_mat = np.ones([1, len(user_item_pair_all)], dtype=bool)
            assert is_obs == True
        else:
            user_item_pair_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])
            live_mat = np.zeros([0, len(indices)], dtype=bool)

        if len(buffer) > 0:
            assert indices is not None
            index = indices
            live_ids = np.ones_like(index, dtype=bool)

            
            while any(live_ids) and len(live_mat) < self.window_size:
                if is_obs: # modeling the obs in the batch
                    user_item_pair = buffer[index].obs
                    rew = buffer.rew_prev[index]
                else: # modeling the obs_next in the batch
                    user_item_pair = buffer[index].obs_next
                    rew = buffer[index].rew
                
                # users = np.expand_dims(user_item_pair[:, 0], -1)
                # items = np.expand_dims(user_item_pair[:, 1], -1)
                
                live_mat = np.vstack([live_mat, live_ids])
                user_item_pair_all = np.concatenate([user_item_pair_all, user_item_pair])
                rew_all = np.concatenate([rew_all, rew])

                dead = buffer.is_start[index]
                live_ids[dead] = False
                index = buffer.prev(index)

        user_all = np.expand_dims(user_item_pair_all[:, 0], -1)
        item_all = np.expand_dims(user_item_pair_all[:, 1], -1)

        e_i = self.get_embedding(item_all, "action")

        if self.use_userEmbedding:
            e_u = self.get_embedding(user_all, "user")
            s_t = torch.cat([e_u, e_i], dim=-1)
        else:
            s_t = e_i

        rew_matrix = rew_all.reshape((-1, 1))
        e_r = self.get_embedding(rew_matrix, "feedback")

        normed_r = self.get_normed_reward(e_r, is_train=is_train)

        if self.reward_handle == "mul":
            state_flat = s_t * normed_r
        elif self.reward_handle == "cat":
            state_flat = torch.cat([s_t, normed_r], 1)
        elif self.reward_handle == "cat2":
            state_flat = torch.cat([s_t, e_r], 1)
        else:
            state_flat = s_t

        # state_flat = s_t
        state_cube = state_flat.reshape((-1, live_mat.shape[1], state_flat.shape[-1]))

        mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
        state_masked = state_cube * mask

        emb_state = torch.swapaxes(state_masked, 0, 1)

        len_states = mask.sum(0).squeeze(-1).cpu().numpy()
        mask = mask.swapaxes(0, 1)

        emb_state_reverse = reverse_padded_sequence(emb_state, len_states)
        seq = emb_state_reverse

        # assert seq.shape[1] == mask.shape[1]
        if seq.shape[1] < self.window_size:
            seq = torch.cat([seq, torch.zeros([seq.shape[0], self.window_size - seq.shape[1], seq.shape[2]], device=self.device)], dim=1)
            mask = torch.cat([mask, torch.zeros([mask.shape[0], self.window_size - mask.shape[1], mask.shape[2]], device=self.device)], dim=1)

        return seq, mask, len_states

