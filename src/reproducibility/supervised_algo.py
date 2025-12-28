from typing import Any

import torch as th
import torch.nn as nn
import torch.optim as optim


class SineNet(nn.Module):
    """
    [1, 64, 64, 1] MLP as used in the Reptile paper.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class SupervisedAlgo:
    """
    wrapper for supervised sine regression so that it
    can use directly into ReptileMetaRL.
    """

    def __init__(self, env, policy: str = "SineNet", **kwargs: Any):
        """
        env: SineTask
        policy: ignored (only SineNet is implemented)
        kwargs: not used here
        """
        self.env = env
        self.policy = SineNet()

    def learn(self,
              total_timesteps: int,
              batch_size: int = 10,
              inner_lr: float = 1e-2,
              **kwargs: Any):
        """
        Treat total_timesteps as "number of gradient steps".

        Each step:
            - sample batch_size points from the SineTask
            - do one Adam step on MSE loss
        """
        self.policy.train()
        optimizer = optim.Adam(self.policy.parameters(), lr=inner_lr)
        loss_fn = nn.MSELoss()

        for _ in range(total_timesteps):
            x, y = self.env.sample(batch_size)
            y_pred = self.policy(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
