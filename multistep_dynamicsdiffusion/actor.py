import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils


class TransformerActor(nn.Module):
    """An isotropic Gaussian policy using Transformer."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        num_layers,
        horizon,
        log_std_bounds,
        num_heads,
    ):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.horizon = horizon
        self.action_dim = action_dim

        self.obs_embedding = nn.Linear(obs_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, horizon, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_mu = nn.Linear(hidden_dim, action_dim)
        self.out_log_std = nn.Linear(hidden_dim, action_dim)

        self.apply(utils.weight_init)

    def forward(self, obs):
        if obs.ndim == 1:  # add batch dimension if needed
            obs = obs.unsqueeze(0)
        if obs.ndim == 2:  # add horizon dimenstion if needed
            obs = obs.unsqueeze(1)
        obs = self.obs_embedding(obs)
        obs = obs + self.positional_encoding  # Add positional encoding

        transformer_output = self.transformer(obs)

        # Get the mean and log standard deviation for each action
        mu = self.out_mu(transformer_output)
        log_std = self.out_log_std(transformer_output)

        # Constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # Sample actions and compute log probabilities
        policy = utils.SquashedNormal(
            mu.view(-1, self.horizon, self.action_dim),
            std.view(-1, self.horizon, self.action_dim),
        )
        pi = policy.rsample()
        log_pi = policy.log_prob(pi).sum(-1, keepdim=True)

        return policy.mean, pi, log_pi
