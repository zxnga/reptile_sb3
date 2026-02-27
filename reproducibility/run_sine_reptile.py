import argparse
import copy
import os
import random
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from src.reptile import ReptileMetaRL
from reproducibility.sine_tasks import SineTaskGenerator, SineTask
from reproducibility.supervised_algo import SupervisedAlgo, SineNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def clone_model(model: th.nn.Module, device: str = "cpu") -> th.nn.Module:
    cloned = copy.deepcopy(model)
    return cloned.to(device)


def make_task_generator(seed: int | None = None) -> SineTaskGenerator:
    return SineTaskGenerator(
        amplitude_range=(0.1, 5.0),
        phase_range=(0.0, 2 * np.pi),
        generator_seed=seed,
    )


def sample_fixed_support(
    task: SineTask,
    k_shots: int,
    device: str = "cpu",
) -> Tuple[th.Tensor, th.Tensor]:
    x, y = task.sample(k_shots)
    return x.to(device), y.to(device)


def dense_grid(
    task: SineTask,
    n_points: int = 50,
    device: str = "cpu",
) -> Tuple[th.Tensor, th.Tensor]:
    x = th.linspace(-5.0, 5.0, n_points, device=device).unsqueeze(1)
    y = task.amplitude * th.sin(x + task.phase)
    return x, y


@th.no_grad()
def mse_on_task(
    model: th.nn.Module,
    task: SineTask,
    n_points: int = 50,
    device: str = "cpu",
) -> float:
    x, y_true = dense_grid(task, n_points=n_points, device=device)
    model = model.to(device)
    model.eval()
    y_pred = model(x)
    return float(((y_pred - y_true) ** 2).mean().item())


def adapt_on_fixed_support(
    init_model: th.nn.Module,
    x_support: th.Tensor,
    y_support: th.Tensor,
    n_steps: int = 32,
    inner_lr: float = 1e-3,
    inner_beta1: float = 0.0,
    inner_beta2: float = 0.999,
    device: str = "cpu",
) -> th.nn.Module:
    adapted = clone_model(init_model, device=device)
    adapted.train()

    x_support = x_support.to(device)
    y_support = y_support.to(device)

    optimizer = th.optim.Adam(
        adapted.parameters(),
        lr=inner_lr,
        betas=(inner_beta1, inner_beta2),
    )
    loss_fn = th.nn.MSELoss()

    for _ in range(n_steps):
        pred = adapted(x_support)
        loss = loss_fn(pred, y_support)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return adapted


def train_reptile_sine_meta(
    outer_steps: int = 2000,
    task_batch_size: int = 5,
    inner_steps: int = 32,
    meta_lr: float = 0.1,
    inner_lr: float = 1e-3,
    k_shots: int = 10,
    hidden_dim: int = 64,
    inner_beta1: float = 0.0,
    inner_beta2: float = 0.999,
    use_meta_optimizer: bool = False,
    seed: int = 0,
    device: str = "cpu",
    verbose: int = 1,
) -> Tuple[ReptileMetaRL, th.nn.Module]:
    meta_learner = ReptileMetaRL(
        tasks_generator_cls=SineTaskGenerator,
        tasks_generator_params={
            "amplitude_range": (0.1, 5.0),
            "phase_range": (0.0, 2 * np.pi),
            "generator_seed": seed,
        },
        rl_algorithm=SupervisedAlgo,
        rl_algo_kwargs={
            "policy": SineNet,
            "device": device,
            "hidden_dim": hidden_dim,
        },
        inner_steps=inner_steps,
        outer_steps=outer_steps,
        task_batch_size=task_batch_size,
        inner_loop_params={
            "batch_size": k_shots,
            "inner_lr": inner_lr,
            "inner_beta1": inner_beta1,
            "inner_beta2": inner_beta2,
            "fixed_support": False,
        },
        meta_lr=meta_lr,
        use_meta_optimizer=use_meta_optimizer,
        meta_optimizer_cls=th.optim.Adam,
        ignored_layers=None,
        verbose=verbose,
        device=device,
        tensorboard_logs=None,
    )

    meta_learner.learn()

    if hasattr(meta_learner, "meta_policy"):
        meta_model = meta_learner.meta_policy
    elif hasattr(meta_learner, "policy"):
        meta_model = meta_learner.policy
    else:
        raise AttributeError("Could not find meta model on ReptileMetaRL.")

    meta_model = meta_model.to(device)
    meta_model.eval()

    return meta_learner, meta_model

def evaluate_initializer(
    init_model_fn: Callable[[], th.nn.Module],
    n_eval_tasks: int = 100,
    n_adapt_steps: int = 32,
    k_shots: int = 10,
    inner_lr: float = 1e-3,
    inner_beta1: float = 0.0,
    inner_beta2: float = 0.999,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[float, float]:
    pre_losses = []
    post_losses = []

    tg = make_task_generator(seed=seed)

    for i in range(n_eval_tasks):
        task, _, _ = tg.get_task(i)
        x_support, y_support = sample_fixed_support(task, k_shots=k_shots, device=device)

        init_model = init_model_fn()
        pre_losses.append(mse_on_task(init_model, task, n_points=50, device=device))

        adapted = adapt_on_fixed_support(
            init_model=init_model,
            x_support=x_support,
            y_support=y_support,
            n_steps=n_adapt_steps,
            inner_lr=inner_lr,
            inner_beta1=inner_beta1,
            inner_beta2=inner_beta2,
            device=device,
        )
        post_losses.append(mse_on_task(adapted, task, n_points=50, device=device))

    return float(np.mean(pre_losses)), float(np.mean(post_losses))

def plot_figure(
    meta_model: th.nn.Module,
    k_shots: int = 10,
    n_adapt_steps: int = 32,
    inner_lr: float = 1e-3,
    hidden_dim: int = 64,
    inner_beta1: float = 0.0,
    inner_beta2: float = 0.999,
    seed: int = 0,
    device: str = "cpu",
    save_dir: str = "./reproducibility/figures",
    file_name: str = "figure1_like_reptile.png",
    show: bool = False,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)

    tg = make_task_generator(seed=seed + 999)
    task, info, _ = tg.get_task(0)

    # one shared fixed support set for both panels
    x_support, y_support = sample_fixed_support(task, k_shots=k_shots, device=device)
    x_plot = th.linspace(-5.0, 5.0, 200, device=device).unsqueeze(1)

    with th.no_grad():
        y_true = task.amplitude * th.sin(x_plot + task.phase)

        random_init = SineNet(hidden_dim=hidden_dim).to(device)
        random_init.eval()
        y_rand_before = random_init(x_plot)

        meta_model = meta_model.to(device)
        meta_model.eval()
        y_meta_before = meta_model(x_plot)

    random_after = adapt_on_fixed_support(
        init_model=random_init,
        x_support=x_support,
        y_support=y_support,
        n_steps=n_adapt_steps,
        inner_lr=inner_lr,
        inner_beta1=inner_beta1,
        inner_beta2=inner_beta2,
        device=device,
    )

    meta_after = adapt_on_fixed_support(
        init_model=meta_model,
        x_support=x_support,
        y_support=y_support,
        n_steps=n_adapt_steps,
        inner_lr=inner_lr,
        inner_beta1=inner_beta1,
        inner_beta2=inner_beta2,
        device=device,
    )

    with th.no_grad():
        y_rand_after = random_after(x_plot)
        y_meta_after = meta_after(x_plot)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    panels = [
        ("Before training", y_rand_before, y_rand_after),
        ("After Reptile training", y_meta_before, y_meta_after),
    ]

    x_plot_np = x_plot.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    x_support_np = x_support.detach().cpu().numpy()
    y_support_np = y_support.detach().cpu().numpy()

    for ax, (title, y_before, y_after) in zip(axes, panels):
        ax.plot(x_plot_np, y_before.detach().cpu().numpy(), linestyle="--", label="Before")
        ax.plot(
            x_plot_np,
            y_after.detach().cpu().numpy(),
            linestyle="-.",
            label=f"After {n_adapt_steps}",
        )
        ax.plot(x_plot_np, y_true_np, label="True")
        ax.scatter(x_support_np, y_support_np, marker="x", label="Sampled")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_xlim(-5.0, 5.0)

    axes[0].set_ylabel("y")

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved figure to: {save_path}")
    print(
        f"Plotted task: amplitude={info['amplitude']:.4f}, "
        f"phase={info['phase']:.4f}, seed={info['seed']}"
    )

    return save_path

def run_reptile_sine_experiment(
    outer_steps: int = 2000,
    task_batch_size: int = 5,
    inner_steps: int = 32,
    meta_lr: float = 0.1,
    inner_lr: float = 1e-3,
    k_shots: int = 10,
    hidden_dim: int = 64,
    inner_beta1: float = 0.0,
    inner_beta2: float = 0.999,
    use_meta_optimizer: bool = False,
    n_eval_tasks: int = 100,
    n_plot_adapt_steps: int = 32,
    seed: int = 0,
    device: str = "cpu",
    save_dir: str = "./reproducibility/figures",
    file_name: str = "figure1_like_reptile.png",
    verbose: int = 1,
    show: bool = False,
) -> Dict[str, object]:
    set_seed(seed)

    if device.startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    _, meta_model = train_reptile_sine_meta(
        outer_steps=outer_steps,
        task_batch_size=task_batch_size,
        inner_steps=inner_steps,
        meta_lr=meta_lr,
        inner_lr=inner_lr,
        k_shots=k_shots,
        hidden_dim=hidden_dim,
        inner_beta1=inner_beta1,
        inner_beta2=inner_beta2,
        use_meta_optimizer=use_meta_optimizer,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    pre_meta, post_meta = evaluate_initializer(
        init_model_fn=lambda: clone_model(meta_model, device=device),
        n_eval_tasks=n_eval_tasks,
        n_adapt_steps=n_plot_adapt_steps,
        k_shots=k_shots,
        inner_lr=inner_lr,
        inner_beta1=inner_beta1,
        inner_beta2=inner_beta2,
        seed=seed + 123,
        device=device,
    )

    pre_rand, post_rand = evaluate_initializer(
        init_model_fn=lambda: SineNet(hidden_dim=hidden_dim).to(device),
        n_eval_tasks=n_eval_tasks,
        n_adapt_steps=n_plot_adapt_steps,
        k_shots=k_shots,
        inner_lr=inner_lr,
        inner_beta1=inner_beta1,
        inner_beta2=inner_beta2,
        seed=seed + 456,
        device=device,
    )

    figure_path = plot_figure(
        meta_model=meta_model,
        k_shots=k_shots,
        n_adapt_steps=n_plot_adapt_steps,
        inner_lr=inner_lr,
        hidden_dim=hidden_dim,
        inner_beta1=inner_beta1,
        inner_beta2=inner_beta2,
        seed=seed,
        device=device,
        save_dir=save_dir,
        file_name=file_name,
        show=show,
    )

    print("\nAverage dense-grid MSE on unseen sine tasks:")
    print(f"  Reptile before adaptation: {pre_meta:.6f}")
    print(f"  Reptile after adaptation : {post_meta:.6f}")
    print(f"  Random  before adaptation: {pre_rand:.6f}")
    print(f"  Random  after adaptation : {post_rand:.6f}")

    return {
        "reptile_pre_mse": pre_meta,
        "reptile_post_mse": post_meta,
        "random_pre_mse": pre_rand,
        "random_post_mse": post_rand,
        "figure_path": figure_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Figure-1-like Reptile sine-wave experiment."
    )

    # training
    parser.add_argument("--outer-steps", type=int, default=2000)
    parser.add_argument("--task-batch-size", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=32)
    parser.add_argument("--meta-lr", type=float, default=0.1)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--k-shots", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)

    # optimizer details
    parser.add_argument("--inner-beta1", type=float, default=0.0)
    parser.add_argument("--inner-beta2", type=float, default=0.999)
    parser.add_argument("--use-meta-optimizer", action="store_true")

    # eval / plotting
    parser.add_argument("--n-eval-tasks", type=int, default=100)
    parser.add_argument("--n-plot-adapt-steps", type=int, default=32)
    parser.add_argument("--save-dir", type=str, default="./reproducibility/figures")
    parser.add_argument("--file-name", type=str, default="figure1_like_reptile.png")
    parser.add_argument("--show", action="store_true")

    # misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = run_reptile_sine_experiment(
        outer_steps=args.outer_steps,
        task_batch_size=args.task_batch_size,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        k_shots=args.k_shots,
        hidden_dim=args.hidden_dim,
        inner_beta1=args.inner_beta1,
        inner_beta2=args.inner_beta2,
        use_meta_optimizer=args.use_meta_optimizer,
        n_eval_tasks=args.n_eval_tasks,
        n_plot_adapt_steps=args.n_plot_adapt_steps,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        file_name=args.file_name,
        verbose=args.verbose,
        show=args.show,
    )

    print("\nFinal results:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
