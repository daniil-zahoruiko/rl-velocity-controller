# Reinforcement Learning Velocity Controller

This project implements a reinforcement learning (RL)-based velocity controller using **Stable-Baselines3** and a **custom OpenAI Gym environment** for an Autonomous Underwater Vehicle with 6 degrees of freedom. The controller is currently trained and evaluated **in simulation only**. See simulation results below. Deployment on the **real robot** is a work in progress. Please note that I will not be publicly releasing full ROS 2 node code.

## 🚀 Overview

The controller learns to map velocity targets and current robot state to thrust commands, replacing traditional PID control with a policy trained through reinforcement learning. This approach enables potentially more adaptive and robust control in dynamic environments.

## 🧠 RL Framework

- **RL Library**: Stable-Baselines3
- **Algorithm**: SAC
- **Environment**: Custom OpenAI Gym-compatible environment simulating robot dynamics
- **Setting**: Currently episodic, with episodes starting and ending on velocity target changes. This might change soon however as I'm considering switching to continuing setting
- **State**: State is a combination of current robot's acceleration and velocity along with previous action error to the target velocity
- **Action**: Action is thrust to the 8 thrusters
- **Reward**: Reward is a weighted combination of error to the target velocity, which is really its main component as it directs the algorithm, with thruster usage and sudden change penalty to make thrusts more smooth and prevent mechanical damage

## 📊 Simulation Results

- The algorithm is able to control any values for 4 degrees of freedom at once - surge, sway, heave, and one of the angular velocities, while maintaining small magnitudes for the 2 other angular velocities to keep the robot stable in those axes.
- Here is an example of an episode where the robot starts still and gains target velocities within 100 episodes, which in simulation represents 1 second of real time:
  
  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/687d8d11-a407-4c7e-9218-37730328b82b" />

  And these are thrusts sent to the thrusters during this episode:

  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/9041a798-f2ab-4b84-9054-bb1ea2509298" />

## 🛠️ Work in Progress

Currently, I am focusing on deploying this on the real robot by using data collected driving around with PID to do off-policy transfer learning on the simulation model.

## 📄 References

Carlucho, I., De Paula, M., Wang, S., Petillot, Y. & Acosta, G. G., 2018. *Adaptive low‑level control of autonomous underwater vehicles using deep reinforcement learning*. *Robotics and Autonomous Systems*, 107, pp. 71–86. doi:10.1016/j.robot.2018.05.016.
