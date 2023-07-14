from core.state_tracker.base import StateTracker_Base
import torch
from torch import nn

FLOAT = torch.FloatTensor

class StateTracker_Caser(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device,
                 use_userEmbedding=False, window_size=10, filter_sizes=[2, 3, 4], num_filters=16,
                 dropout_rate=0.1):
        super().__init__(user_columns=user_columns, action_columns=action_columns,
                         feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                         window_size=window_size)

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        self.hidden_size = action_columns[0].embedding_dim
        self.num_item = action_columns[0].vocabulary_size

        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.final_dim = self.hidden_size + self.num_filters_total

        # Item embedding
        embedding_dict = torch.nn.ModuleDict(
            {"feat_item": torch.nn.Embedding(num_embeddings=self.num_item + 1,
                                             embedding_dim=self.hidden_size)})
        self.embedding_dict = embedding_dict.to(device)
        nn.init.normal_(self.embedding_dict.feat_item.weight, mean=0, std=0.1)

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


    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, **kwargs):

        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices, obs, reset, is_obs)

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