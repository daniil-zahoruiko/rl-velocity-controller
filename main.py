import logging

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EveryNTimesteps
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


register(
    id="pool_train",
    entry_point="env:PoolEnvTrain",
)


class PlotCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self):
        self.model.get_env().env_method("plot_last_episode", n=self.n_calls)

        return True


"""
class PolicyCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.log_file = "/at_everything/ws/src/at_control_rl/at_control_rl/exp.log"
        logging.basicConfig(
            filename=self.log_file, level=logging.INFO, format="%(message)s", filemode="w"
        )

    def _on_step(self) -> bool:
        obs = {
            key: torch.tensor(val)
            for (key, val) in self.training_env.env_method("_get_obs")[0].items()
        }
        mean_actions, log_std, _ = self.model.actor.get_action_dist_params(obs)

        log_entry = {"step": self.num_timesteps, "policy": list(zip(mean_actions, log_std))}
        logging.info(log_entry)

        return True
"""


def linear_schedule(initial_value: float, final_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value

    return func

def train():
    env = gym.make("pool_train")
    vec_env = VecNormalize(
        DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False
    )
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch={'pi': [600, 400, 300], 'qf': [600, 400, 300]})
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(8, ), sigma=1.0, theta=0.3)
    model = SAC(
        "MultiInputPolicy",
        env,
        # action_noise=action_noise,
        verbose=1,
        learning_rate=linear_schedule(0.003, 0.0001),
        # tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=policy_kwargs,
    )
    print(model.policy)

    total_timesteps = 350_000
    plot_callback = EveryNTimesteps(n_steps=total_timesteps // 20, callback=PlotCallback())
    # policy_callback = EveryNTimesteps(n_steps=total_timesteps // 1000, callback=PolicyCallback())

    # Combine them in a list
    callback_list = CallbackList([plot_callback])  # policy_callback])

    # Start training
    model.learn(total_timesteps=total_timesteps, callback=callback_list)
    model.save("model.zip")


# from train import train

if __name__ == "__main__":
    train()
