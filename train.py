from networks import ActorNetwork, CriticNetwork
from OU import OU
from dynamics import Dynamics
import numpy as np
import torch
import random

torch.autograd.set_detect_anomaly(True)

def train(episodes, episode_steps):
    num_thrusters = 8
    max_velocity = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 0.7], dtype=np.float32)
    period = 0.1
    min_buf_size = 4
    max_buf_size = 200000
    beta = 0.001
    gamma = 0.99
    N=2
    replay_buffer = []

    actor = ActorNetwork(num_thrusters=num_thrusters).to('cuda')
    critic = CriticNetwork(num_thrusters=num_thrusters).to('cuda')
    target_actor = ActorNetwork(num_thrusters=num_thrusters).to('cuda')
    target_critic = CriticNetwork(num_thrusters=num_thrusters).to('cuda')

    critic_loss_fn = torch.nn.MSELoss()
    critic_optimizer = torch.optim.Adam(critic.parameters())

    actor_optimizer = torch.optim.Adam(actor.parameters())

    noise = OU(scale=0.02, mean=np.zeros(shape=(num_thrusters, ), dtype=np.float32), variance=0.09)
    for e in range(episodes):

        noise.reset()
        dynamics = get_dynamics()

        target = torch.from_numpy(np.random.uniform(-max_velocity, max_velocity)).to(torch.float32).to('cuda')
        # state is velocities, accelerations, last action, errors 
        state = np.zeros(shape=(18 + num_thrusters, ), dtype=np.float32)
        state[-6:] = np.copy(target.cpu())
        state = torch.from_numpy(state).to('cuda')
        for t in range(episode_steps):
            predicted_action = actor(state)

            if len(replay_buffer) > min_buf_size:
                transitions = random.sample(replay_buffer, N)
                targets = torch.tensor([], requires_grad=True).to('cuda')
                outputs = torch.tensor([], requires_grad=True).to('cuda')
                for transition in transitions:
                    next_action = target_actor(transition[3])
                    target_val = transition[2] + gamma * target_critic(torch.cat((transition[3], next_action)))
                    output_val = critic(torch.cat((transition[0], transition[1])))
                    targets = torch.cat((targets, target_val)).to('cuda')
                    outputs = torch.cat((outputs, output_val)).to('cuda')

                critic_loss = critic_loss_fn(outputs, targets)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic(torch.cat((state, predicted_action)))
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                def soft_update(target_net, source_net):
                    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                        target_param.data.copy_(beta * source_param.data + (1 - beta) * target_param.data)

                soft_update(target_actor, actor)
                soft_update(target_critic, critic)

            
            action = predicted_action.detach() + noise.sample().to('cuda')

            dynamics.target_thrust = action.cpu().numpy()
            dynamics.update(period)

            velocity = torch.from_numpy(dynamics.velocity).to(torch.float32).to('cuda')
            next_state = torch.concatenate([velocity, torch.from_numpy(dynamics.acceleration).to(torch.float32).to('cuda'), action, target - velocity])
            replay_buffer.append((state, action, calculate_reward(next_state, num_thrusters), next_state))
            if len(replay_buffer) > max_buf_size:
                replay_buffer.pop(0)
            state = next_state

        print(state[:6]) 
        print(target)       

def calculate_reward(state, num_thrusters):
    velocity = state[:6]
    error = state[-6:]
    action = state[-(6 + num_thrusters):]

    alpha = 1
    scales = torch.ones((6,)).to('cuda')  # TODO: check if this is correct
    square_error = error @ torch.diag(scales) @ error
    # thruster_usage = np.sum(np.abs(action))
    # sudden_change_penalty = np.linalg.norm(
    #     np.mean(self.sliding_window[: min(self.sliding_window_size, self.step_cnt + 1)], axis=0)
    #     - action
    # )
    reward = (
        torch.exp(-1 / (alpha**2) * square_error)
        # - self.zeta * thruster_usage
        # - self.xi * sudden_change_penalty
    )
    return reward

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
    water_level: float = 0.0

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
        np.zeros(6),  # TODO: Check if this is correct
        np.zeros(6),
    )
    return dynamics