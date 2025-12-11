import torch as th
from reptile_agent import ReptileAgent
from sine_tasks import SineTaskGenerator, SineTask
from supervised_algo import SupervisedAlgo, SineNet
import math
import numpy as np


def main():

    outer_steps = 2000 # meta-iterations
    task_batch_size = 5 # tasks per meta-iteration
    inner_steps = 5 # gradient steps per task during meta-training
    meta_lr = 0.1 # Reptile step size ε
    inner_lr = 1e-2 # inner-loop learning rate
    k_shots = 10 # batch_size per inner step

    agent = ReptileAgent(
        tasks_generator_cls=SineTaskGenerator,
        tasks_generator_params={},       # default amp range
        inner_steps=inner_steps,
        outer_steps=outer_steps,
        meta_lr=meta_lr,
        rl_algorithm=SupervisedAlgo,
        re_use_actors=False,             # not needed
        actor_layers=[],                 # ignored because re_use_actors=False
        use_actor_meta_weights=True,
        task_batch_size=task_batch_size,
        rl_algo_kwargs={},               # no SB3-specific kwargs
        ignored_layers=[],               # learn all params
        use_meta_optimizer=False,        # direct Reptile update
        inner_loop_params={
            "batch_size": k_shots,
            "inner_lr": inner_lr,
        },
        split_rollout_updates=True,      # irrelevant here
        save_frequency=500,
        meta_weights_dir="./meta_sine_weights",
        tensorboard_logs=None,
        experience_name="reptile_sine"
    )

    meta_algo = agent.train()
    meta_init = meta_algo.policy

    # eval
    def mse_on_task(model: SineNet, task: SineTask, n_points: int = 50) -> float:
        x = th.linspace(-5.0, 5.0, n_points).unsqueeze(1)
        with th.no_grad():
            y_true = task.amplitude * th.sin(x + task.phase)
            y_pred = model(x)
            mse = ((y_pred - y_true) ** 2).mean().item()
        return mse

    def evaluate(meta_model: SineNet,
                 n_eval_tasks: int = 20,
                 n_adapt_steps: int = 32):
        pre_losses = []
        post_losses = []

        for _ in range(n_eval_tasks):
            # sample new task
            from sine_tasks import SineTaskGenerator
            tg = SineTaskGenerator()
            task, _, _ = tg.get_task(0)

            # before adaptation
            pre_loss = mse_on_task(meta_model, task)
            pre_losses.append(pre_loss)

            # copy meta model and adapt on this task
            adapted = SineNet()
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

    pre, post = evaluate(meta_init)
    print(f"Average MSE before adaptation: {pre:.4f}")
    print(f"Average MSE after 32 steps:    {post:.4f}")


if __name__ == "__main__":
    main()
