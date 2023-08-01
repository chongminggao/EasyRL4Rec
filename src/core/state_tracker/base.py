# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 9:49 上午
# @Author  : Chongming GAO
# @FileName: state_tracker.py

import numpy as np
import torch
from torch import nn, Tensor

from core.util.inputs import input_from_feature_columns
from core.userModel.user_model import build_input_features
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
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device="cpu", window_size=10):
        super().__init__()
        self.user_columns = user_columns
        self.action_columns = action_columns
        self.feedback_columns = feedback_columns

        self.user_index = build_input_features(user_columns)
        self.action_index = build_input_features(action_columns)
        self.feedback_index = build_input_features(feedback_columns)

        self.dim_model = dim_model
        self.window_size = window_size
        self.device = device

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

    def build_state(self, obs=None,
                    env_id=None,
                    obs_next=None,
                    reset=False, **kwargs):
        if reset:
            self.user = None
            return

        if obs is not None:  # 1. initialize the state vectors
            self.user = obs
            # item = np.ones_like(obs) * np.nan
            item = np.ones_like(obs) * self.num_item
            ui_pair = np.hstack([self.user, item])
            res = {"obs": ui_pair}

        elif obs_next is not None:  # 2. add action autoregressively
            item = obs_next
            user = self.user[env_id]
            ui_pair = np.hstack([user, item])
            res = {"obs_next": ui_pair}

        return res
    
    def convert_to_k_state_embedding(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None):
        if reset:  # get user embedding
            # users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            # a = self.get_embedding(items_window, "action")

            e_i = self.get_embedding(items, "action")
            emb_state = e_i.repeat_interleave(self.window_size, dim=0).reshape([len(e_i), self.window_size, -1])
            seq = emb_state

            len_states = np.ones([len(emb_state)], dtype=int)
            mask = torch.zeros([seq.shape[0], seq.shape[1], 1], device=self.device)
            mask[:, 0, :] = 1

            seq = seq * mask

            return seq, mask, len_states

        else:
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)

            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])
            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True
            '''
                Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
                Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            # while not all(flag_has_init) and len(live_mat) < self.window_size:
            while len(live_mat) < self.window_size:
                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init  # just dead and have not been initialized before.
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[~live_id_prev, 1] = self.num_item
                rew_prev[~live_id_prev] = 1
                # obs_prev[ind_init, 1] = self.num_item
                # rew_prev[ind_init] = 1
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])

                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            # user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")

            s_t = e_i

            state_flat = s_t
            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            emb_state = torch.swapaxes(state_masked, 0, 1)

            len_states = mask.sum(0).squeeze(-1).cpu().numpy()
            mask = mask.swapaxes(0, 1)

            emb_state_reverse = reverse_padded_sequence(emb_state, len_states)

            seq = emb_state_reverse

            return seq, mask, len_states




