# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn


class TransformerCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, n_emb, num_heads, num_layers):
        super().__init__()

        self.act_embed = nn.Linear(action_dim, n_emb)
        self.obs_embed = nn.Linear(obs_dim, n_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=num_heads, batch_first=True, norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out = nn.Linear(n_emb, 1)

    def forward(self, init_obs, actions):
        # init_obs: (batch_size, obs_dim)
        # actions: (batch_size, horizon, action_dim)

        actions = self.act_embed(actions)
        init_obs = self.obs_embed(init_obs)

        if init_obs.dim() == 2:
            init_obs = init_obs.unsqueeze(1)  # (batch_size, 1, n_emb)

        x = torch.cat([init_obs, actions], dim=1)  # (batch_size, horizon + 1, n_emb)

        encoded = self.encoder(x)  # (batch_size, horizon + 1, n_emb)

        output = self.out(encoded)  # (batch_size, horizon + 1, 1)

        # Summation because we want to predict the total reward as Q value function
        output = output[:, 1:, :]  # (batch_size, horizon)

        return output


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, n_emb, num_heads, num_layers):
        super().__init__()

        self.Q1 = TransformerCritic(obs_dim, action_dim, n_emb, num_heads, num_layers)
        self.Q2 = TransformerCritic(obs_dim, action_dim, n_emb, num_heads, num_layers)

        self.outputs = dict()

    def forward(self, obs, actions):
        """
        Forward pass of the critic network.

        Args:
            obs (torch.Tensor): Input observations.
            actions (torch.Tensor): Input actions.

        Returns:
            torch.Tensor: Q-value estimates for the given observations and actions.
        """

        assert obs.size(0) == actions.size(0)

        q1 = self.Q1(obs, actions)
        q2 = self.Q2(obs, actions)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2
