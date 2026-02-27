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
    Wrapper for supervised sine regression so it can be used inside ReptileMetaRL.
    """

    def __init__(
        self,
        env,
        policy=SineNet,
        device: str = "cpu",
        hidden_dim: int = 64,
        **kwargs: Any,
    ):
        """
        env: SineTask
        policy: model class (default SineNet)
        """
        self.env = env
        self.device = device
        self.policy = policy(hidden_dim=hidden_dim).to(device)

    def learn(
        self,
        total_timesteps: int,
        batch_size: int = 10,
        inner_lr: float = 1e-2,
        inner_beta1: float = 0.0,
        inner_beta2: float = 0.999,
        fixed_support: bool = True,
        **kwargs: Any,
    ):
        """
        Treat total_timesteps as 'number of gradient steps'.

        If fixed_support=True:
            - sample ONE batch from the sine task
            - reuse it for all inner updates

        If fixed_support=False:
            - resample a new batch every step
        """
        self.policy.train()

        optimizer = optim.Adam(
            self.policy.parameters(),
            lr=inner_lr,
            betas=(inner_beta1, inner_beta2),
        )
        loss_fn = nn.MSELoss()

        x_fixed = y_fixed = None
        if fixed_support:
            x_fixed, y_fixed = self.env.sample(batch_size)
            x_fixed = x_fixed.to(self.device)
            y_fixed = y_fixed.to(self.device)

        for _ in range(total_timesteps):
            if fixed_support:
                x, y = x_fixed, y_fixed
            else:
                x, y = self.env.sample(batch_size)
                x = x.to(self.device)
                y = y.to(self.device)

            y_pred = self.policy(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self
