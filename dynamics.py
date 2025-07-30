# Author: Sergii Penner

from math import cos, sin, tan

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

# forwards: Force (N) = T200_FORWARD_RPM_TO_THRUST * RPM^2
T200_FORWARD_RPM_TO_THRUST = 4.48e-6
# reverse: Force (N) = T200_REVERSE_RPM_TO_THRUST * RPM^2
T200_REVERSE_RPM_TO_THRUST = -3.78e-6
T200_MIN_RPM = 150
T200_MAX_RPM = 3400
# maximum rate the thrusters can change RPM (RPM/s)
T200_RPM_CHANGE_RATE = 10000

ARCTOS_MAX_VELOCITY = np.array([0.5, 0.5, 0.5, 0.7, 0.7, 0.45])


def S(x: NDArray) -> NDArray:
    """Skew symmetric matrix."""
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])  # [1](2.10)


def linear_transformation_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Linear transformation matrix."""
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()  # [1](2.18)


def angular_transformation_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Angular transformation matrix."""
    return np.array(
        [
            [1, sin(roll) * tan(pitch), cos(roll) * tan(pitch)],
            [0, cos(roll), -sin(roll)],
            [0, sin(roll) / cos(pitch), cos(roll) / cos(pitch)],
        ]
    )  # [1](2.28)


class Dynamics:
    """Dynamics class for simulating the underwater physics of the robot."""

    def __init__(
        self,
        gravity: float,
        water_density: float,
        water_level: float,
        mass: float,
        center_of_mass: NDArray[np.float64],
        moments_of_inertia: NDArray[np.float64],
        products_of_inertia: NDArray[np.float64],
        displacement: float,
        displacement_radius: float,
        center_of_buoyancy: NDArray[np.float64],
        damping: NDArray[np.float64],
        quadratic_damping: NDArray[np.float64],
        mass_ratio: float,
        thruster_positions: NDArray[np.float64],
        thruster_directions: NDArray[np.float64],
        thruster_taus: NDArray[np.float64],
        velocity_noise: NDArray[np.float64],
        initial_pose: NDArray[np.float64],
        initial_velocity: NDArray[np.float64],
        disable_hydrostatic_forces: bool = False,
    ) -> None:
        """
        Initialize the dynamics.

        :param gravity: Gravity.
        :param water_density: Water density.
        :param water_level: Water level in NED world frame.
        :param mass: Mass of the robot.
        :param center_of_mass: Center of mass of the robot in body frame.
        :param moments_of_inertia: Moments of inertia of the robot in body frame.
        :param products_of_inertia: Products of inertia of the robot in body frame.
        :param displacement: Displacement of the robot.
        :param displacement_radius: Displacement radius of the robot.
        :param center_of_buoyancy: Center of buoyancy of the robot in body frame.
        :param damping: Damping of the robot.
        :param quadratic_damping: Quadratic damping of the robot.
        :param mass_ratio: mass_ratio = added mass / mass.
        :param thruster_positions: Positions of the thrusters in body frame.
        :param thruster_directions: Directions of the thrusters in body frame.
        :param thruster_taus: Thruster reaction delays. (time to reach full thrust?)
        :param velocity_noise: Noise added to the velocity.
        :param initial_pose: Initial pose of the robot in NED world frame.
        :param initial_velocity: Initial velocity of the robot in body frame.
        :param disable_hydrostatic_forces: Disable hydrostatic forces.
        """
        self.gravity = gravity
        self.water_density = water_density
        self.water_level = water_level
        self.mass = mass
        self.center_of_mass = center_of_mass.copy()
        self.moments_of_inertia = moments_of_inertia.copy()
        self.products_of_inertia = products_of_inertia.copy()
        self.displacement = displacement
        self.displacement_radius = displacement_radius
        self.center_of_buoyancy = center_of_buoyancy.copy()
        self.damping = damping.copy()
        self.quadratic_damping = quadratic_damping.copy()
        self.mass_ratio = mass_ratio
        self.num_thrusters = thruster_positions.shape[0]
        self.thruster_positions = thruster_positions.copy()
        self.thruster_directions = thruster_directions.copy()
        self.thruster_taus = thruster_taus.copy()
        self.velocity_noise = velocity_noise.copy()
        self.disable_hydrostatic_forces = disable_hydrostatic_forces

        self.mass_matrix = self.mass_rigid_body_matrix() * (1 + self.mass_ratio)  # [1](6.48)
        self.inverse_mass_matrix = np.linalg.inv(self.mass_matrix)

        # NED position and orientation of the robot as: [x, y, z, φ, θ, ψ]
        self.pose = initial_pose.copy()

        Rnb = linear_transformation_matrix(*self.pose[3:6])
        Tnb = angular_transformation_matrix(*self.pose[3:6])

        # linear and angular acceleration of the robot in the body frame as: [u, v, w, p, q, r]
        self.acceleration = np.zeros(6)

        # linear and angular velocity of the robot in the body frame as: [u, v, w, p, q, r]
        self.velocity = initial_velocity.copy()
        # NED linear and angular velocity of the robot in the world frame
        self.world_velocity = np.concatenate((Rnb @ self.velocity[:3], Tnb @ self.velocity[3:6]))

        # target thrust of the thrusters
        self.target_thrust = np.zeros(self.num_thrusters)
        # current rpm of the thrusters
        self.rpm = np.zeros(self.num_thrusters)

    def inertia_matrix(self) -> NDArray[np.float64]:
        """Moment of inertia matrix about CO."""
        return np.array(
            [
                [
                    self.moments_of_inertia[0],
                    self.products_of_inertia[0],
                    self.products_of_inertia[1],
                ],
                [
                    self.products_of_inertia[0],
                    self.moments_of_inertia[1],
                    self.products_of_inertia[2],
                ],
                [
                    self.products_of_inertia[1],
                    self.products_of_inertia[2],
                    self.moments_of_inertia[2],
                ],
            ]
        )  # [1](3.1)

    def mass_rigid_body_matrix(self) -> NDArray[np.float64]:
        """Rigid body system inertia matrix."""
        return np.block(
            [
                [self.mass * np.eye(3), -self.mass * S(self.center_of_mass)],
                [self.mass * S(self.center_of_mass), self.inertia_matrix()],
            ]
        )  # [1](3.44)

    def coriolis_matrix(self, velocity: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Coriolis-Centripetal matrix.

        :param velocity: Linear and angular velocity of the robot in body frame.
        """
        s1 = S(
            self.mass_matrix[0:3, 0:3] @ velocity[0:3] + self.mass_matrix[0:3, 3:6] @ velocity[3:6]
        )
        s2 = S(
            self.mass_matrix[3:6, 0:3] @ velocity[0:3] + self.mass_matrix[3:6, 3:6] @ velocity[3:6]
        )
        return np.block([[np.zeros((3, 3)), -s1], [-s1, -s2]])  # [1](3.46)

    def damping_matrix(self, velocity: NDArray[np.float64]) -> NDArray[np.float64]:
        """Linear and quadratic damping matrix."""
        return -np.diag(self.damping + self.quadratic_damping * np.abs(velocity))

    def hydrostatic_forces(self, pose: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Gravity and buoyancy forces.

        :param pose: NED position and orientation of the robot in world frame.
        """
        W = self.mass * self.gravity  # weight

        # estimate displacement on sphere model
        # float from 0 to 2 of how many radii of the object are underwater
        radius_ratio_underwater = (
            max(-1.0, min(1.0, (pose[2] - self.water_level) / self.displacement_radius)) + 1.0
        )
        # approximate volume ratio from 0 to 1 of how much of the object is underwater based on a spherical cap
        volume_ratio_underwater = radius_ratio_underwater**2 * (3.0 - radius_ratio_underwater) / 4.0
        B = (
            self.water_density * self.gravity * self.displacement * volume_ratio_underwater
        )  # buoyancy

        # gravity and buoyancy centers in the body frame
        gx, gy, gz = self.center_of_mass
        bx, by, bz = self.center_of_buoyancy

        # restoring force and moment vector in the body frame [1](4.6)
        g = np.array(
            [
                (W - B) * sin(pose[4]),
                -(W - B) * cos(pose[4]) * sin(pose[3]),
                -(W - B) * cos(pose[4]) * cos(pose[3]),
                -(gy * W - by * B) * cos(pose[4]) * cos(pose[3])
                + (gz * W - bz * B) * cos(pose[4]) * sin(pose[3]),
                (gz * W - bz * B) * sin(pose[4]) + (gx * W - bx * B) * cos(pose[4]) * cos(pose[3]),
                -(gx * W - bx * B) * cos(pose[4]) * sin(pose[3]) - (gy * W - by * B) * sin(pose[4]),
            ]
        )

        return g

    def thruster_force(self, thrust: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Force and torque from thrusters.

        :param thrust: Current thrust of each thruster.
        """
        B = np.zeros((6, self.num_thrusters))
        B[0:3, :] = self.thruster_directions.T
        B[3:6, :] = np.cross(self.thruster_positions, self.thruster_directions).T

        return B @ thrust

    def rpm_to_thrust(self, rpm: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert RPM to thrust.

        :param rpm: RPM of each thruster.
        """
        thrust = np.zeros(self.num_thrusters)
        for i in range(self.num_thrusters):
            if rpm[i] >= 0.0:
                thrust[i] = T200_FORWARD_RPM_TO_THRUST * rpm[i] ** 2
            else:
                thrust[i] = T200_REVERSE_RPM_TO_THRUST * rpm[i] ** 2

        return thrust

    def thrust_to_rpm(self, thrust: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert thrust to RPM.

        :param thrust: Thrust of each thruster.
        """
        rpm = np.zeros(self.num_thrusters)
        for i in range(self.num_thrusters):
            if thrust[i] >= 0.0:
                rpm[i] = np.sqrt(thrust[i] / T200_FORWARD_RPM_TO_THRUST)
            else:
                rpm[i] = -np.sqrt(thrust[i] / T200_REVERSE_RPM_TO_THRUST)

        return rpm

    def thruster_dynamics(self, dt: float):
        """
        Update thrust from target thrust.

        :param dt: Time step.
        """
        target_rpm = self.thrust_to_rpm(self.target_thrust)
        # set target rpm to 0 if its absolute value is less than T200_MIN_RPM
        target_rpm[np.abs(target_rpm) < T200_MIN_RPM] = 0.0
        # clip target rpm to T200_MAX_RPM
        target_rpm = np.clip(target_rpm, -T200_MAX_RPM, T200_MAX_RPM)

        self.rpm += np.clip(
            target_rpm - self.rpm, -T200_RPM_CHANGE_RATE * dt, T200_RPM_CHANGE_RATE * dt
        )

    def inverse_dynamics(
        self, pose: NDArray[np.float64], velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute acceleration from forces using equation [1](1.21).

        :param pose: NED position and orientation of the robot in world frame.
        :param velocity: Linear and angular velocity of the robot in body frame.
        """
        tau = self.thruster_force(self.rpm_to_thrust(self.rpm))
        c_v = self.coriolis_matrix(velocity) @ velocity
        d_v = self.damping_matrix(velocity) @ velocity
        if self.disable_hydrostatic_forces:
            g = np.zeros(6)
        else:
            g = self.hydrostatic_forces(pose)
        return self.inverse_mass_matrix @ (tau - d_v - c_v - g)  # acceleration

    def update(self, dt: float) -> None:
        """
        Update the dynamics by the given time step.

        :param dt: Time step.
        """
        Rnb = linear_transformation_matrix(*self.pose[3:6])
        Tnb = angular_transformation_matrix(*self.pose[3:6])

        self.thruster_dynamics(dt)
        self.acceleration = self.inverse_dynamics(self.pose, self.velocity)
        self.velocity += (
            self.acceleration + np.random.uniform(-1, 1, 6) * self.velocity_noise
        ) * dt
        self.world_velocity = np.concatenate((Rnb @ self.velocity[0:3], Tnb @ self.velocity[3:6]))
        self.pose += self.world_velocity * dt


"""
References:
- [1] Handbook of Marine Craft Hydrodynamics and Motion Control by Thor I. Fossen (2011)
"""
