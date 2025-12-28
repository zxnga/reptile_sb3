import torch as th
import os
from torch import nn, optim
from src.reptile import ReptileMetaRL
from src.reproducibility.sine_tasks import SineTaskGenerator, SineTask
from src.reproducibility.supervised_algo import SupervisedAlgo, SineNet
import math
import numpy as np
import matplotlib.pyplot as plt

def adapt_on_task(
    init_model: nn.Module,
    task: SineTask,
    k_shots: int = 10,
    n_adapt_steps: int = 32,
    inner_lr: float = 1e-2,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> nn.Module:
    """
    Copy init_model and run a few gradient steps on (x, y) from a single sine task.
    Returns the adapted model.
    """
    adapted = SineNet(hidden_dim=hidden_dim).to(device)
    adapted.load_state_dict(init_model.state_dict())

    optimizer = optim.Adam(adapted.parameters(), lr=inner_lr)
    loss_fn = nn.MSELoss()

    adapted.train()
    for _ in range(n_adapt_steps):
        x, y = task.sample(k_shots)  # sample from SineTask

        y_pred = adapted(x.to(device))
        loss = loss_fn(y_pred, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return adapted


def mse_on_task(model: SineNet, task: SineTask, n_points: int = 50) -> float:
    x = th.linspace(-5.0, 5.0, n_points).unsqueeze(1)
    with th.no_grad():
        y_true = task.amplitude * th.sin(x + task.phase)
        y_pred = model(x)
        mse = ((y_pred - y_true) ** 2).mean().item()
    return mse


def evaluate_meta_init(meta_model: SineNet,
             n_eval_tasks: int = 20,
             n_adapt_steps: int = 32,
             k_shots: int = 10,
             inner_lr: float = 1e-2) -> tuple[float, float]:
    pre_losses = []
    post_losses = []

    for _ in range(n_eval_tasks):
        tg = SineTaskGenerator(
            amplitude_range=(0.1, 5.0),
            phase_range=(0.0, 2 * np.pi),
        )
        task, _, _ = tg.get_task(0)

        # before adaptation
        pre_loss = mse_on_task(meta_model, task)
        pre_losses.append(pre_loss)

        adapted = SineNet(hidden_dim=64)
        adapted.load_state_dict(meta_model.state_dict())

        optimizer = th.optim.Adam(adapted.parameters(), lr=inner_lr)
        loss_fn = th.nn.MSELoss()

        for _ in range(n_adapt_steps):
            x, y = task.sample(k_shots)
            y_pred = adapted(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        post_loss = mse_on_task(adapted, task)
        post_losses.append(post_loss)

    return float(np.mean(pre_losses)), float(np.mean(post_losses))

def evaluate_random_init(n_eval_tasks: int = 20,
                         n_adapt_steps: int = 32,
                         k_shots: int = 10,
                         inner_lr: float = 1e-2) -> tuple[float, float]:
    pre_losses = []
    post_losses = []

    for _ in range(n_eval_tasks):
        tg = SineTaskGenerator(
            amplitude_range=(0.1, 5.0),
            phase_range=(0.0, 2 * np.pi),
        )
        task, _, _ = tg.get_task(0)

        # fresh random init
        model = SineNet(hidden_dim=64)

        pre_losses.append(mse_on_task(model, task))

        adapted = SineNet(hidden_dim=64)
        adapted.load_state_dict(model.state_dict())

        opt = th.optim.Adam(adapted.parameters(), lr=inner_lr)
        loss_fn = th.nn.MSELoss()

        for _ in range(n_adapt_steps):
            x, y = task.sample(k_shots)
            y_pred = adapted(x)
            loss = loss_fn(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        post_losses.append(mse_on_task(adapted, task))

    return float(np.mean(pre_losses)), float(np.mean(post_losses))


def plot_meta_sine_with_random(
    meta_init: nn.Module,
    title: str = "Reptile vs Random on sine tasks",
    n_tasks: int = 4,
    k_shots: int = 10,
    n_adapt_steps: int = 32,
    inner_lr: float = 1e-2,
    hidden_dim: int = 64,
    device: str = "cpu",
    save_dir: str = "./figures",
    file_name: str = "reptile_vs_random_sine.png",
    show: bool = False,
):
    """
    For each sampled sine task, plot:
      - True function
      - Reptile init before adaptation
      - Reptile init after adaptation
      - Random init before adaptation
      - Random init after adaptation
      - K-shot training samples

    Also prints average MSEs over tasks for Reptile vs Random (before/after).
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)

    meta_init = meta_init.to(device)
    meta_init.eval()

    tg = SineTaskGenerator(
        amplitude_range=(0.1, 5.0),
        phase_range=(0.0, 2 * th.pi),
    )

    x_plot = th.linspace(-5.0, 5.0, 200).unsqueeze(1).to(device)

    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 3), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    # For numeric comparison
    reptile_pre_mses = []
    reptile_post_mses = []
    random_pre_mses = []
    random_post_mses = []

    for i, ax in enumerate(axes):
        # new task
        task, info, _ = tg.get_task(i)
        amp = info["amplitude"]
        phase = info["phase"]

        # ground truth
        with th.no_grad():
            y_true = amp * th.sin(x_plot + phase)

        # ---------------- Reptile init ----------------
        with th.no_grad():
            y_meta_init = meta_init(x_plot)

        adapted_meta = adapt_on_task(
            meta_init,
            task,
            k_shots=k_shots,
            n_adapt_steps=n_adapt_steps,
            inner_lr=inner_lr,
            hidden_dim=hidden_dim,
            device=device,
        )

        with th.no_grad():
            y_meta_adapted = adapted_meta(x_plot)

        # ---------------- Random init -----------------
        random_init = SineNet(hidden_dim=hidden_dim).to(device)
        random_init.eval()
        with th.no_grad():
            y_rand_init = random_init(x_plot)

        adapted_rand = adapt_on_task(
            random_init,
            task,
            k_shots=k_shots,
            n_adapt_steps=n_adapt_steps,
            inner_lr=inner_lr,
            hidden_dim=hidden_dim,
            device=device,
        )

        with th.no_grad():
            y_rand_adapted = adapted_rand(x_plot)

        # ---------------- Training samples ------------
        x_train, y_train = task.sample(k_shots)

        # ---------------- Plotting --------------------
        ax.plot(x_plot.cpu().numpy(), y_true.cpu().numpy(), label="True function")
        ax.plot(
            x_plot.cpu().numpy(),
            y_meta_init.cpu().numpy(),
            linestyle="--",
            label="Reptile before",
        )
        ax.plot(
            x_plot.cpu().numpy(),
            y_meta_adapted.cpu().numpy(),
            linestyle="-.",
            label=f"Reptile after {n_adapt_steps}",
        )
        ax.plot(
            x_plot.cpu().numpy(),
            y_rand_init.cpu().numpy(),
            linestyle=":",
            label="Random before",
        )
        ax.plot(
            x_plot.cpu().numpy(),
            y_rand_adapted.cpu().numpy(),
            linestyle="-",
            alpha=0.7,
            label=f"Random after {n_adapt_steps}",
        )
        ax.scatter(x_train.numpy(), y_train.numpy(), marker="x", label="K-shot samples")

        ax.set_title(f"Task {i+1}\na={amp:.2f}, b={phase:.2f}")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")

        # ---------------- MSEs on dense grid ----------
        mse = lambda y_hat: float(((y_hat - y_true) ** 2).mean().item())
        reptile_pre_mses.append(mse(y_meta_init))
        reptile_post_mses.append(mse(y_meta_adapted))
        random_pre_mses.append(mse(y_rand_init))
        random_post_mses.append(mse(y_rand_adapted))

    handles, labels = axes[0].get_legend_handles_labels()
    # deduplicate legend entries
    uniq = dict(zip(labels, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.20),
    )
    fig.suptitle(title, y=1.28)
    fig.tight_layout()

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved sine meta-learning figure to: {save_path}")

    # Print average dense-grid MSEs over the n_tasks shown
    print("\nDense-grid MSE over plotted tasks (x in [-5, 5]):")
    print(f"  Reptile  before: {np.mean(reptile_pre_mses):.4f}")
    print(f"  Reptile  after : {np.mean(reptile_post_mses):.4f}")
    print(f"  Random   before: {np.mean(random_pre_mses):.4f}")
    print(f"  Random   after : {np.mean(random_post_mses):.4f}")


def main():
    # not specified in paper
    outer_steps = 2000 # meta-iterations
    task_batch_size = 5 # tasks per meta-iteration
    inner_steps = 5 # gradient steps per task during meta-training
    meta_lr = 1e-3 # Reptile learning rate
    inner_lr = 1e-2 # inner-loop learning rate
    k_shots = 10 # batch_size per inner step

    meta_learner = ReptileMetaRL(
        tasks_generator_cls=SineTaskGenerator,
        tasks_generator_params={
            "amplitude_range": (0.1, 5.0),
            "phase_range": (0.0, 2 * np.pi),
        },
        rl_algorithm=SupervisedAlgo,
        rl_algo_kwargs={
            "policy": SineNet,
            "device": "cpu",
            "hidden_dim": 64,
        },
        inner_steps=inner_steps,  
        outer_steps=outer_steps,
        task_batch_size=task_batch_size,
        inner_loop_params={
            "batch_size": k_shots,
            "inner_lr": inner_lr,
        },
        meta_lr=meta_lr,
        use_meta_optimizer=True,
        meta_optimizer_cls=th.optim.Adam,
        ignored_layers=None, 
        # save_frequency=500,
        verbose=1,
        device="cpu",
        tensorboard_logs=None, 
    )

    meta_learner.learn()
    meta_init = meta_learner.policy 

    plot_meta_sine_with_random(
        meta_init=meta_init,
        n_tasks=4,
        k_shots=10,
        n_adapt_steps=32,
        inner_lr=1e-2,
        hidden_dim=64,
        device="cpu",
        save_dir="./src/reproducibility/figures",
        file_name="reptile_vs_random_sine_optimizer.png",
        show=False,
    )

if __name__ == "__main__":
    main()