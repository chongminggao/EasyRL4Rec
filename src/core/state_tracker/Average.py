from core.state_tracker.base import StateTracker_Base
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.util.utils import compute_input_dim

FLOAT = torch.FloatTensor


class StateTrackerAvg(StateTracker_Base):
    def __init__(self, 
                 user_columns, action_columns, feedback_columns, 
                 dim_model,
                 train_max=None, 
                 train_min=None, 
                 test_max=None, 
                 test_min=None, 
                 reward_handle=None,
                 saved_embedding=None,
                 device="cpu", 
                 use_userEmbedding=False, window_size=10):
        
        assert saved_embedding is not None
        super(StateTrackerAvg, self).__init__(
            user_columns=user_columns, 
            action_columns=action_columns,
            feedback_columns=feedback_columns,
            dim_model=dim_model,
            train_max=train_max,
            train_min=train_min,
            test_max=test_max,
            test_min=test_min,
            reward_handle=reward_handle,
            saved_embedding=saved_embedding,
            device=device,
            window_size=window_size, 
            use_userEmbedding=use_userEmbedding)

        self.final_dim = self.hidden_size

        # self.test_min = test_min
        # self.test_max = test_max
        # self.train_min = train_min
        # self.train_max = train_max
        # self.reward_handle = reward_handle

        # assert saved_embedding is not None
        # self.embedding_dict = saved_embedding.to(device)

        # self.num_item = self.embedding_dict.feat_item.weight.shape[0]
        # self.dim_item = self.embedding_dict.feat_item.weight.shape[1]

        # # Add a new embedding vector
        # new_embedding = FLOAT(1, self.dim_model).to(device)
        # nn.init.normal_(new_embedding, mean=0, std=0.01)
        # emb_cat = torch.cat([self.embedding_dict.feat_item.weight.data, new_embedding])
        # new_item_embedding = torch.nn.Embedding.from_pretrained(
        #     emb_cat, freeze=not self.embedding_dict.feat_item.weight.requires_grad)
        # self.embedding_dict.feat_item = new_item_embedding

        # if self.use_userEmbedding:
        #     self.ffn_user = nn.Linear(compute_input_dim(self.user_columns), self.dim_model, device=self.device)


    def forward(self, buffer=None, indices=None, is_obs=None, batch=None, is_train=True, use_batch_in_statetracker=False, **kwargs):

        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices=indices, is_obs=is_obs, batch=batch, use_batch_in_statetracker=use_batch_in_statetracker, is_train=is_train)

        state_sum = seq.sum(dim=1)
        state_final = state_sum / torch.from_numpy(np.expand_dims(len_states, -1)).to(self.device)
        
        return state_final


    # def forward(self, buffer, indices, is_obs=True, batch=None, is_train=True, use_batch_in_statetracker=False):
        
    #     if use_batch_in_statetracker: # when collector collects the data, batch is not None.
    #         assert batch is not None
    #         user_item_pair_all = batch.obs
    #         rew_all = batch.rew_prev
    #         live_mat = np.ones([1, len(user_item_pair_all)], dtype=bool)
    #         assert is_obs == True
    #     else:
    #         user_item_pair_all = np.zeros([0, 2], dtype=int)
    #         rew_all = np.zeros([0])
    #         live_mat = np.zeros([0, len(indices)], dtype=bool)

    #     if len(buffer) > 0:
    #         assert indices is not None
    #         index = indices
    #         live_ids = np.ones_like(index, dtype=bool)

            
    #         while any(live_ids) and len(live_mat) < self.window_size:
    #             if is_obs: # modeling the obs in the batch
    #                 user_item_pair = buffer[index].obs
    #                 rew = buffer.rew_prev[index]
    #             else: # modeling the obs_next in the batch
    #                 user_item_pair = buffer[index].obs_next
    #                 rew = buffer[index].rew
                
    #             # users = np.expand_dims(user_item_pair[:, 0], -1)
    #             # items = np.expand_dims(user_item_pair[:, 1], -1)
                
    #             live_mat = np.vstack([live_mat, live_ids])
    #             user_item_pair_all = np.concatenate([user_item_pair_all, user_item_pair])
    #             rew_all = np.concatenate([rew_all, rew])

    #             dead = buffer.is_start[index]
    #             live_ids[dead] = False
    #             index = buffer.prev(index)

    #     user_all = np.expand_dims(user_item_pair_all[:, 0], -1)
    #     item_all = np.expand_dims(user_item_pair_all[:, 1], -1)

    #     e_i = self.get_embedding(item_all, "action")

    #     if self.use_userEmbedding:
    #         e_u = self.get_embedding(user_all, "user")
    #         s_t = torch.cat([e_u, e_i], dim=-1)
    #     else:
    #         s_t = e_i

    #     rew_matrix = rew_all.reshape((-1, 1))
    #     e_r = self.get_embedding(rew_matrix, "feedback")

    #     if is_train:
    #         r_max = self.train_max
    #         r_min = self.train_min
    #     else:
    #         r_max = self.test_max
    #         r_min = self.test_min

    #     if r_max is not None and r_min is not None:
    #         normed_r = (e_r - r_min) / (r_max - r_min)
    #         # if not (all(normed_r<=1) and all(normed_r>=0)):
    #         #     a = 1
    #         normed_r[normed_r>1] = 1 # todo: corresponding to the initialize reward line above.
    #         # assert (all(normed_r<=1) and all(normed_r>=0))
    #     else:
    #         normed_r = e_r

    #     if self.reward_handle == "mul":
    #         state_flat = s_t * normed_r
    #     elif self.reward_handle == "cat":
    #         state_flat = torch.cat([s_t, normed_r], 1)
    #     elif self.reward_handle == "cat2":
    #         state_flat = torch.cat([s_t, e_r], 1)
    #     else:
    #         state_flat = s_t

    #     state_cube = state_flat.reshape((-1, live_mat.shape[1], state_flat.shape[-1]))

    #     mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
    #     state_masked = state_cube * mask

    #     state_sum = state_masked.sum(dim=0)
    #     state_final = state_sum / torch.from_numpy(np.expand_dims(live_mat.sum(0), -1)).to(self.device)


    #     return state_final


