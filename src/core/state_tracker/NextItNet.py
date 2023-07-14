import torch
import torch.nn as nn
import torch.nn.functional as F

from core.state_tracker2 import StateTracker_Base, extract_axis_1


class StateTracker_NextItNet(StateTracker_Base):
    def __init__(
        self,
        user_columns,
        action_columns,
        feedback_columns,
        dim_model,
        device,
        use_userEmbedding=False,
        window_size=10,
        dilations="[1, 2, 1, 2, 1, 2]",
    ):
        super().__init__(
            user_columns=user_columns,
            action_columns=action_columns,
            feedback_columns=feedback_columns,
            dim_model=dim_model,
            device=device,
            window_size=window_size,
        )
        self.hidden_size = action_columns[0].embedding_dim
        self.num_item = action_columns[0].vocabulary_size

        self.dilations = eval(dilations)
        self.device = device
    
        embedding_dict = torch.nn.ModuleDict(
            {"feat_item": torch.nn.Embedding(
            num_embeddings=self.num_item + 1,
            embedding_dim=self.hidden_size,
        )})
        self.embedding_dict = embedding_dict.to(device)

        # Initialize embedding
        nn.init.normal_(self.embedding_dict.feat_item.weight, 0, 0.01)

        # Convolutional Layers
        self.cnns = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=1,
                    residual_channels=self.hidden_size,
                    kernel_size=3,
                    dilation=i,
                    hidden_size=self.hidden_size,
                )
                for i in self.dilations
            ]
        )
        self.final_dim = self.hidden_size
        # self.s_fc = nn.Linear(self.hidden_size, self.num_item)

    def forward(
        self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, **kwargs
    ):
        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices, obs, reset, is_obs)
        # seq *= mask
        conv_out = seq
        for cnn in self.cnns:
            conv_out = cnn(conv_out)
            conv_out *= mask
        state_final = extract_axis_1(conv_out, len_states - 1).squeeze(1)

        # supervised_output = self.s_fc(state_hidden).squeeze()
        return state_final


class VerticalCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, hidden_size):
        super(VerticalCausalConv, self).__init__()

        # attributes:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.dilation = dilation
        self.hidden_size = hidden_size
        assert out_channels == hidden_size

        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, hidden_size),
            dilation=(dilation, 1),
        )

    def forward(self, seq):
        seq = F.pad(seq, pad=[0, 0, (self.kernel_size - 1) * self.dilation, 0])
        conv2d_out = self.conv2d(seq)
        return conv2d_out


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, residual_channels, kernel_size, dilation, hidden_size
    ):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.hidden_size = hidden_size
        self.residual_channels = residual_channels
        assert (
            residual_channels == hidden_size
        )  # In order for output to be the same size

        self.conv1 = VerticalCausalConv(
            in_channels=in_channels,
            out_channels=residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            hidden_size=hidden_size,
        )
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.conv2 = VerticalCausalConv(
            in_channels=in_channels,
            out_channels=residual_channels,
            kernel_size=kernel_size,
            dilation=dilation * 2,
            hidden_size=hidden_size,
        )
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def forward(self, input_):
        input_unsqueezed = input_.unsqueeze(1)
        conv1_out = self.conv1(input_unsqueezed).permute(0, 3, 2, 1)

        ln1_out = self.ln1(conv1_out)
        relu1_out = F.relu(ln1_out)

        conv2_out = self.conv2(relu1_out).permute(0, 3, 2, 1)
        ln2_out = self.ln2(conv2_out)
        relu2_out = F.relu(ln2_out)
        relu2_out = relu2_out.squeeze()

        out = input_ + relu2_out
        return out
