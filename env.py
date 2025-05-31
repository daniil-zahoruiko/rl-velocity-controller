import os

import gymnasium as gym
import numpy as np
from dynamics import Dynamics
from matplotlib import pyplot as plt


def get_dynamics():
    robot = "arctos"

    gravity = 9.81
    water_density = 1.0

    mass = 38.0
    displacement = 39.0
    displacement_radius = 0.3

    center_of_mass = np.array([-0.0011, -0.0004, -0.0369], dtype=np.float64)
    center_of_buoyancy = np.array([-0.0001, -0.0003, -0.0571], dtype=np.float64)

    moments_of_inertia = np.array([2.0690, 1.6031, 2.3423], dtype=np.float64)
    products_of_inertia = np.array([-0.0094, -0.0734, -0.0079], dtype=np.float64)

    # Xu, Yv, Zw, Kp, Mq, Nr
    damping = np.array([-100.0, -100.0, -200.0, -20.0, -20.0, -20.0], dtype=np.float64)
    # Xuu, Yvv, Zww, Kpp, Mqq, Nrr
    quadratic_damping = np.array([-100.0, -100.0, -200.0, -10.0, -10.0, -10.0], dtype=np.float64)

    # mass_ratio = added mass / mass
    mass_ratio = 0.5

    thruster_positions = np.array(
        [
            0.008,
            -0.272,
            -0.052,
            0.008,
            0.272,
            -0.052,
            0.000,
            -0.002,
            -0.238,
            0.000,
            -0.008,
            0.200,
            -0.251,
            -0.236,
            0.042,
            -0.251,
            0.236,
            0.042,
            0.251,
            -0.236,
            0.042,
            0.251,
            0.236,
            0.042,
        ],
        dtype=np.float64,
    ).reshape((-1, 3))

    # 3D unit vector (i,j,k) for thruster orientation
    # a positive value indicates the direction of net force caused by thrust
    thruster_directions = np.array(
        [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
        ]
    ).reshape((-1, 3))

    # thruster reaction delay
    thruster_taus = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)

    # random noise added to the velocity
    velocity_noise = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2], dtype=np.float64)
    water_level: float = -1000.0

    dynamics = Dynamics(
        gravity,
        water_density,
        water_level,
        mass,
        center_of_mass,
        moments_of_inertia,
        products_of_inertia,
        displacement,
        displacement_radius,
        center_of_buoyancy,
        damping,
        quadratic_damping,
        mass_ratio,
        thruster_positions,
        thruster_directions,
        thruster_taus,
        velocity_noise,
        np.zeros(6),
        np.zeros(6),
    )
    return dynamics


class PoolEnvTrain(gym.Env):

    def __init__(self, max_steps=400, lambda_=0.5, zeta=0.05, xi=0.2, n_sim_steps_per_action=1, total_timesteps=350_000):
        self.period = 1 / 100
        self.max_steps = max_steps
        self.total_timesteps = total_timesteps
        self.step_cnt = 0
        self.total_step_cnt = 0
        self.num_thrusters = 8
        self.max_velocity = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 0.7], dtype=np.float32)
        self.n_sim_steps_per_action = n_sim_steps_per_action
        self.sliding_window_size = 1
        self.dynamics = None
        self.target_state = np.zeros(6)
        self.lambda_ = lambda_
        self.zeta = zeta
        self.xi = xi
        self.observation_space = gym.spaces.Dict(
            {
                "acceleration": gym.spaces.Box(low=-5, high=5, shape=(6,)),
                "velocities": gym.spaces.Box(
                    low=-self.max_velocity, high=self.max_velocity, shape=(6,)
                ),
                "errors": gym.spaces.Box(low=-2 * self.max_velocity, high=2 * self.max_velocity, shape=(6,)),
                "last_action": gym.spaces.Box(
                    low=-1, high=1, shape=(self.num_thrusters,)
                )
            }
        )

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_thrusters,))

        self.last_episode = {}
        self.curr_episode_velocities = np.empty((0, 6))
        self.curr_episode_thrusts = np.empty((0, 8))

        self.dynamics = get_dynamics()

    def _get_obs(self):
        return {
            "acceleration": np.array(self.dynamics.acceleration, dtype=np.float32),
            "velocities": np.array(self.dynamics.velocity, dtype=np.float32),
            "errors": np.array(self.dynamics.velocity, dtype=np.float32)
            - np.array(self.target_state, dtype=np.float32),
            "last_action": np.array(self.last_action, dtype=np.float32)
        }

    
    def _get_reward(self, action):
        velocity_error = (self.dynamics.velocity - self.target_state) / self.max_velocity
        reward = -self.lambda_ * np.sum(np.abs(velocity_error)) - self.zeta * np.sum(np.abs(action)) - self.xi * np.sum(np.abs(action - self.last_action))
        return reward
    

    def _get_info(self):
        return {}

    def step(self, action):
        thrust = action * 60
        self.dynamics.target_thrust = thrust

        self.curr_episode_velocities = np.vstack(
            (self.curr_episode_velocities, self.dynamics.velocity)
        )

        self.curr_episode_thrusts = np.vstack((self.curr_episode_thrusts, thrust))

        for _ in range(self.n_sim_steps_per_action):
            self.dynamics.update(self.period)

        observation = self._get_obs()
        terminated = False
        truncated = self.step_cnt > self.max_steps
        reward = self._get_reward(action)
        info = self._get_info()

        self.sliding_window[self.step_cnt % self.sliding_window_size] = action
        self.last_action = action

        self.step_cnt += 1
        self.total_step_cnt += 1

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_episode["target"] = self.target_state
        self.last_episode["velocities"] = self.curr_episode_velocities
        self.last_episode["thrusts"] = self.curr_episode_thrusts
        self.curr_episode_velocities = np.empty((0, 6))
        self.dynamics = get_dynamics()
        self.step_cnt = 0
        self.sliding_window = np.zeros((self.sliding_window_size, self.num_thrusters))
        self.last_action = np.zeros((self.num_thrusters,))
        self.target_state = np.random.uniform(-self.max_velocity, self.max_velocity)
        if self.total_step_cnt < self.total_timesteps // 5:
            self.target_state[3:] = 0.0
        elif self.total_step_cnt < 2 * self.total_timesteps // 5:
            self.target_state[3:] = 0.0
            idx = np.random.choice([3, 4, 5], size=1)
            self.target_state[idx] = np.random.uniform(-0.1, 0.1)
        elif self.total_step_cnt < 3 * self.total_timesteps // 5:
            self.target_state[3:] = np.random.uniform(-0.1, 0.1)
        else:
            idx = np.random.choice([3, 4, 5], size=2, replace=False)
            self.target_state[idx] = np.random.uniform(-0.1, 0.1)
            if self.total_step_cnt >= 4 * self.total_timesteps // 5:
                self.dynamics.velocity = np.random.uniform(-self.max_velocity, self.max_velocity)
                idx = np.random.choice([3, 4, 5], size=2, replace=False)
                self.dynamics.velocity[idx] = np.random.uniform(-0.1, 0.1)

        obs = self._get_obs()
        return obs, self._get_info()

    def plot_last_episode(self, n):
        print("Plotting...")
        target_vs_velocity_path = "plots/target_vs_velocity"
        thrust_path = "plots/thrust"

        if not os.path.exists(target_vs_velocity_path):
            os.makedirs(target_vs_velocity_path)
        if not os.path.exists(thrust_path):
            os.makedirs(thrust_path)

        fig, axs = plt.subplots(2, 3)
        x = np.arange(self.last_episode["velocities"].shape[0])
        titles = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]

        for i, ax in enumerate(axs.flat):
            ax.plot(x, self.last_episode["velocities"][:, i], label="Velocity")
            ax.axhline(self.last_episode["target"][i], color="r", linestyle="--", label="Target",)
            ax.set_ylim([-1, 1])
            ax.set_title(titles[i])

        handles, labels = axs.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels) 

        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(os.path.join(target_vs_velocity_path, f"plot{n}.png"))

        fig, axs = plt.subplots(2, self.num_thrusters // 2)
        axs.flatten()

        thrust_entries = self.max_steps
        x = np.arange(thrust_entries)
        for i, ax in enumerate(axs.flat):
            ax.plot(x, self.last_episode["thrusts"][-thrust_entries:, i])
            ax.set_ylim([-60, 60])
            ax.set_title(f"Thrust {i}")
        
        fig.tight_layout()
        plt.savefig(os.path.join(thrust_path, f"plot{n}.png"))
