from core.state_tracker.base import StateTracker_Base
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.userModel.utils import compute_input_dim

FLOAT = torch.FloatTensor


class StateTrackerAvg(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, saved_embedding,
                 train_max=None, train_min=None, test_max=None, test_min=None, reward_handle="no",
                 device="cpu", use_userEmbedding=False, window_size=10):
        super(StateTrackerAvg, self).__init__(user_columns=user_columns, action_columns=action_columns,
                                               feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                                               window_size=window_size)

        self.test_min = test_min
        self.test_max = test_max
        self.train_min = train_min
        self.train_max = train_max
        self.reward_handle = reward_handle

        assert saved_embedding is not None
        self.embedding_dict = saved_embedding.to(device)

        self.num_item = self.embedding_dict.feat_item.weight.shape[0]
        self.dim_item = self.embedding_dict.feat_item.weight.shape[1]

        # Add a new embedding vector
        new_embedding = FLOAT(1, self.dim_model).to(device)
        nn.init.normal_(new_embedding, mean=0, std=0.01)
        emb_cat = torch.cat([self.embedding_dict.feat_item.weight.data, new_embedding])
        new_item_embedding = torch.nn.Embedding.from_pretrained(
            emb_cat, freeze=not self.embedding_dict.feat_item.weight.requires_grad)
        self.embedding_dict.feat_item = new_item_embedding

        self.use_userEmbedding = use_userEmbedding
        if self.use_userEmbedding:
            self.ffn_user = nn.Linear(compute_input_dim(self.user_columns), self.dim_model, device=self.device)

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, is_train=True):

        if reset:  # get user embedding

            users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            # e_i = torch.ones(obs.shape[0], self.dim_item, device=self.device)
            # nn.init.normal_(e_i, mean=0, std=0.0001)

            e_i = self.get_embedding(items, "action")

            if self.use_userEmbedding:
                e_u = self.get_embedding(users, "user")
                # s0 = self.ffn_user(e_u)
                s0 = torch.cat([e_u, e_i], dim=-1)
            else:
                s0 = e_i

            r0 = torch.ones(len(s0), 1).to(s0.device) # todo: define init reward as 1
            if self.reward_handle == "mul":
                state_res = s0 * 1
            elif self.reward_handle == "cat":
                state_res = torch.cat([s0, r0], 1)
            elif self.reward_handle == "cat2":
                state_res = torch.cat([s0, r0], 1)
            else:
                state_res = s0

            return state_res

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
            while not all(flag_has_init) and len(live_mat) < self.window_size:
                # if not remove_recommended_ids and len(live_mat) >= self.window_size:
                #     break

                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[ind_init, 1] = self.num_item  # initialize obs
                rew_prev[ind_init] = 1  # todo: initialize reward.
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])
                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            # item_all_complete = np.expand_dims(obs_all[:, 1], -1)
            if len(live_mat) > self.window_size:
                live_mat = live_mat[:self.window_size, :]
                obs_all = obs_all[:len(index) * self.window_size, :]
                rew_all = rew_all[:len(index) * self.window_size]

            user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")

            rew_matrix = rew_all.reshape((-1, 1))
            e_r = self.get_embedding(rew_matrix, "feedback")

            if self.use_userEmbedding:
                e_u = self.get_embedding(user_all, "user")
                s_t = torch.cat([e_u, e_i], dim=-1)
            else:
                s_t = e_i

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

            if self.reward_handle == "mul":
                state_flat = s_t * normed_r
            elif self.reward_handle == "cat":
                state_flat = torch.cat([s_t, normed_r], 1)
            elif self.reward_handle == "cat2":
                state_flat = torch.cat([s_t, e_r], 1)
            else:
                state_flat = s_t

            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            state_sum = state_masked.sum(dim=0)
            state_final = state_sum / torch.from_numpy(np.expand_dims(live_mat.sum(0), -1)).to(self.device)

            # if remove_recommended_ids:
            #     recommended_ids = item_all_complete.reshape(-1, len(index)).T
            #     return state_final, recommended_ids
            # else:
            #     return state_final, None

            return state_final