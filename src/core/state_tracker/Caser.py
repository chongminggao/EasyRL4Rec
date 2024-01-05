from src.core.state_tracker.base import StateTracker_Base
import torch
from torch import nn

FLOAT = torch.FloatTensor

class StateTracker_Caser(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns,
                 dim_model,
                 train_max=None, 
                 train_min=None, 
                 test_max=None, 
                 test_min=None, 
                 reward_handle=None,
                 saved_embedding=None,
                 device="cpu",
                 use_userEmbedding=False, window_size=10, filter_sizes=[2, 3, 4], num_filters=16,
                 dropout_rate=0.1):
        super().__init__(
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

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.final_dim = self.hidden_size + self.num_filters_total

        

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.window_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, buffer=None, indices=None, is_obs=None, batch=None, is_train=True, use_batch_in_statetracker=False, **kwargs):

        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices=indices, is_obs=is_obs, batch=batch, use_batch_in_statetracker=use_batch_in_statetracker, is_train=is_train)

        emb_state_final = seq.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(emb_state_final))
            h_out = h_out.squeeze(-1)
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(emb_state_final))
        v_flat = v_out.view(-1, self.hidden_size)

        state_hidden = torch.cat([h_pool_flat, v_flat], 1)
        state_hidden_dropout = self.dropout(state_hidden)

        return state_hidden_dropout