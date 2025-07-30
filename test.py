import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium.envs.registration import register
import time

register(
    id="pool_train",
    entry_point="env:PoolEnvTrain",
)

model = SAC.load("model.zip")
env   = gym.make("pool_train")

n_episodes = 10
rewards = []
inference_times = []
for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        start_time = time.time()
        action, _ = model.predict(obs, deterministic=True)
        inference_times.append(time.time() - start_time)
        obs, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += r
    print(f"Episode {ep} reward = {ep_reward:.3f}")
    rewards.append(ep_reward)
    if ep > 0:
        env.unwrapped.plot_last_episode(f'_test_{ep}')

env.close()

print(f"\nMean reward over {n_episodes} eps = {sum(rewards)/n_episodes:.3f}")
print(f"\nAverage inference time: {sum(inference_times) / len(inference_times)}")