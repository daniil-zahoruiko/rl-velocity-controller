import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="pool_train",
    entry_point="env:PoolEnvTrain",
)

model = SAC.load("model_good.zip")
env   = gym.make("pool_train")

n_episodes = 10
rewards = []
for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += r
    print(f"Episode {ep} reward = {ep_reward:.3f}")
    rewards.append(ep_reward)
    if ep > 0:
        env.unwrapped.plot_last_episode(f'_test_{ep}')

print(f"\nMean reward over {n_episodes} eps = {sum(rewards)/n_episodes:.3f}")
env.close()