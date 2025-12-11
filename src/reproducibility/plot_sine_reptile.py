import torch as th
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.reproducibility.sine_tasks import SineTaskGenerator
from src.reproducibility.supervised_algo import SineNet


def adapt_on_task(init_model: nn.Module,
                  task,
                  k_shots: int = 10,
                  n_adapt_steps: int = 32,
                  inner_lr: float = 1e-2) -> nn.Module:
    """
    Copy init_model and run a few gradient steps on (x, y) from a single task.
    Returns the adapted model.
    """
    adapted = SineNet()
    adapted.load_state_dict(init_model.state_dict())

    optimizer = optim.Adam(adapted.parameters(), lr=inner_lr)
    loss_fn = nn.MSELoss()

    adapted.train()
    for _ in range(n_adapt_steps):
        x, y = task.sample(k_shots)
        y_pred = adapted(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return adapted


def plot_meta_sine(
    init_model: nn.Module,
    title: str = "Reptile meta-initialization",
    n_tasks: int = 4,
    k_shots: int = 10,
    n_adapt_steps: int = 32,
    inner_lr: float = 1e-2,
):
    """
    Produce a figure similar in spirit to the Reptile paper's sine plots:
      - Several columns (tasks)
      - Each column: ground truth sine, init prediction, post-adaptation prediction,
        and the K samples used for adaptation.
    """
    tg = SineTaskGenerator()

    x_plot = th.linspace(-5.0, 5.0, 200).unsqueeze(1)

    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 3), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # sample new task
        task, info, _ = tg.get_task(i)
        amp = info["amplitude"]
        phase = info["phase"]

        # ground truth
        with th.no_grad():
            y_true = amp * th.sin(x_plot + phase)

            y_init = init_model(x_plot)

        # adapt from the same init on this task
        adapted = adapt_on_task(
            init_model,
            task,
            k_shots=k_shots,
            n_adapt_steps=n_adapt_steps,
            inner_lr=inner_lr,
        )

        with th.no_grad():
            y_adapted = adapted(x_plot)

        # also show the actual training points used (one batch for visualization)
        x_train, y_train = task.sample(k_shots)

        ax.plot(x_plot.numpy(), y_true.numpy(), label="True function")
        ax.plot(x_plot.numpy(), y_init.numpy(), linestyle="--", label="Before adaptation")
        ax.plot(x_plot.numpy(), y_adapted.numpy(), linestyle="-.", label=f"After {n_adapt_steps} steps")
        ax.scatter(x_train.numpy(), y_train.numpy(), marker="x", label="K-shot samples")

        ax.set_title(f"Task {i+1}\na={amp:.2f}, b={phase:.2f}")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.15))
    fig.suptitle(title, y=1.25)
    fig.tight_layout()
    plt.show()
