from networks import ActorNetwork, CriticNetwork
from OU import OU
from dynamics import Dynamics
import numpy as np

def train(episodes, episode_steps):
    num_thrusters = 8
    max_velocity = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 0.7], dtype=np.float32)

    actor = ActorNetwork(num_thrusters=num_thrusters)
    critic = CriticNetwork(num_thrusters=num_thrusters)
    goal_actor = ActorNetwork(num_thrusters=num_thrusters)
    goal_critic = CriticNetwork(num_thrusters=num_thrusters)

    noise = OU(scale=0.02, mean=np.zeros(shape=(num_thrusters, )), variance=0.09)
    for e in range(episodes):

        noise.reset()
        dynamics = get_dynamics()

        target = np.random.uniform(-max_velocity, max_velocity)
        # state is velocities, accelerations, last action, errors 
        state = np.zeros(shape=(18 + num_thrusters, ))
        state[-6:] = np.copy(target)
        for t in range(episode_steps):
            pass

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