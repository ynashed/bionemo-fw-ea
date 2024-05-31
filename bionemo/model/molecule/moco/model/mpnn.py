import math

import torch
import torch.nn as nn
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "gelu_tanh": nn.GELU(approximate='tanh'),
    "sigmoid": nn.Sigmoid(),
}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        activation: str = 'silu',
        dropout: float = 0.0,
    ):
        """
        Initialize the MLP.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_size (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output features.
            num_hidden_layers (int): Number of hidden layers.
            activation (str): Activation function to use ('relu', 'silu', etc.).
            dropout (float): Dropout probability (between 0 and 1).
        """
        super(MLP, self).__init__()

        if activation not in NONLINEARITIES:
            raise ValueError(f"Activation function must be one of {list(NONLINEARITIES.keys())}")

        self.act_layer = NONLINEARITIES[activation]

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(NONLINEARITIES[activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(NONLINEARITIES[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))

        # Combine all layers into a sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaLN(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int):
        """
        Initialize the Adaptive Layer Normalization (AdaLN) module.
        This implementation does not learn a gate.

        Args:
            condition_dim (int): Dimension of the conditional input.
            feature_dim (int): Dimension of the input features.
        """
        super().__init__()
        self.layernorm = BatchLayerNorm(feature_dim)
        self.scale_shift_mlp = MLP(condition_dim, 2 * feature_dim, 2 * feature_dim)

    def forward(self, h: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AdaLN module.

        Args:
            h (torch.Tensor): Input tensor to be normalized (batch_size, feature_dim).
            t (torch.Tensor): Conditional input tensor (batch_size, condition_dim).

        Returns:
            torch.Tensor: Normalized output tensor (batch_size, feature_dim).
        """
        # scale = 1 + self.scale_mlp(t)
        # shift = self.shift_mlp(t)
        scale, shift = self.scale_shift_mlp(t).chunk(2, dim=-1)
        return (1 + scale[batch]) * self.layernorm(h, batch) + shift[batch]


class EGNN(nn.Module):
    def __init__(self, invariant_feature_dim, equivariant_feature_dim):
        self.time_embedder = TimestepEmbedder(invariant_feature_dim)
        self.adaLN = AdaLN(invariant_feature_dim, invariant_feature_dim)
        self.message_input_size = 0
        self.h_input_size = 0
        self.phi_message = MLP(self.message_input_size, invariant_feature_dim, invariant_feature_dim)
        self.message_gate = MLP(invariant_feature_dim, invariant_feature_dim, 1)
        self.phi_x = MLP(invariant_feature_dim, invariant_feature_dim, equivariant_feature_dim)
        self.phi_h = MLP(self.h_input_size, invariant_feature_dim, invariant_feature_dim)
        self.self.node_agg_scale = 1  # /100
        self.coord_change_maximum = 10
        # TODO: Understand how to do all to all message passing use EqGatDiff as a reference https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/experiments/diffusion_discrete_addfeats.py#L43
        # ! This is the underlying MPNN they use EQGATGlobalEdgeConvFinal

    def coord2radial(self, coord, edge_index):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))
        return radial, coord_diff

    def coord2diff(self, x, edge_index, norm=False, norm_constant=1):
        row, col = edge_index
        coord_diff = x[row] - x[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        norm = torch.sqrt(radial + 1e-8)
        if norm:
            coord_diff = coord_diff / (norm + norm_constant)
        return radial, coord_diff

    def forward(self, h, x, edge_index, edge_attr, mask):
        src, dst = edge_index
        hi, hj = h[src], h[dst]
        radial, coord_diff = self.coord2radial(x, edge_index)
        radial = radial.reshape(radial.shape[0], -1)
        mij = self.phi_message(torch.cat([hi, hj, radial, edge_attr], -1))
        gij = self.message_gate(mij)
        h_agg = unsorted_segment_sum(mij * gij, src, num_segments=x.size(0))
        h = h + self.phi_h(torch.cat([x, h_agg, h], -1))
        trans = coord_diff * self.phi_x(mij).unsqueeze(-1)
        x_agg = unsorted_segment_mean(trans, src, num_segments=x.size(0))
        x = x + x_agg.clamp(-self.coord_change_maximum, self.coord_change_maximum)
        return h, x


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result
