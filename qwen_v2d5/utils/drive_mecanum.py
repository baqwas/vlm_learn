# -----------------------------------------------------------------------------
# four_wheel_robot_sim.py
#
# A Python script to simulate the movement of a four-wheel mecanum drive
# robot on a 2D plane. The script demonstrates omnidirectional movement
# by using inverse kinematics to control the robot.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import math


def simulate_mecanum_movement(start_x, start_y, start_theta, wheel_radius, robot_width, robot_length, desired_v_x,
                              desired_v_y, desired_omega, duration, dt):
    """
    Simulates the movement of a four-wheel mecanum drive robot based on desired
    linear and angular velocities.

    Args:
        start_x (float): Initial x-coordinate of the robot.
        start_y (float): Initial y-coordinate of the robot.
        start_theta (float): Initial orientation of the robot in radians.
        wheel_radius (float): Radius of the mecanum wheels.
        robot_width (float): The width of the robot from wheel to wheel.
        robot_length (float): The length of the robot from wheel to wheel.
        desired_v_x (float): Desired linear velocity in the x-direction (m/s).
        desired_v_y (float): Desired linear velocity in the y-direction (m/s).
        desired_omega (float): Desired angular velocity (rad/s).
        duration (float): Total simulation time in seconds.
        dt (float): Time step for the simulation.

    Returns:
        tuple: A tuple containing lists of x, y, and theta values over time.
    """
    # Initialize lists to store the robot's path and orientation
    x_history = [start_x]
    y_history = [start_y]
    theta_history = [start_theta]

    # Initialize current position and orientation
    current_x = start_x
    current_y = start_y
    current_theta = start_theta

    # Kinematic constant combining half-width and half-length
    L = (robot_length / 2) + (robot_width / 2)

    # Inverse Kinematics Matrix (maps robot velocity to wheel velocities)
    # The matrix columns correspond to [v_x, v_y, omega]
    inverse_kinematics_matrix = np.array([
        [1, -1, -L],  # Front Left Wheel
        [1, 1, L],  # Front Right Wheel
        [1, 1, -L],  # Rear Left Wheel
        [1, -1, L]  # Rear Right Wheel
    ])

    # Desired velocities in a vector
    desired_velocities = np.array([[desired_v_x], [desired_v_y], [desired_omega]])

    # Calculate required wheel speeds (rad/s)
    # wheel_speeds = np.dot(inverse_kinematics_matrix, desired_velocities) / wheel_radius

    # Simpler calculation without matrix multiplication for clarity
    v_fl = (desired_v_x - desired_v_y - desired_omega * L)
    v_fr = (desired_v_x + desired_v_y + desired_omega * L)
    v_rl = (desired_v_x + desired_v_y - desired_omega * L)
    v_rr = (desired_v_x - desired_v_y + desired_omega * L)

    # The simulation loop iterates over the specified duration
    for _ in np.arange(0, duration, dt):
        # Update the position and orientation of the robot
        # The change in position is based on the current orientation (theta)
        d_x = (desired_v_x * math.cos(current_theta) - desired_v_y * math.sin(current_theta)) * dt
        d_y = (desired_v_x * math.sin(current_theta) + desired_v_y * math.cos(current_theta)) * dt
        d_theta = desired_omega * dt

        # Update the robot's position and orientation
        current_x += d_x
        current_y += d_y
        current_theta += d_theta

        # Store the new state in the history lists
        x_history.append(current_x)
        y_history.append(current_y)
        theta_history.append(current_theta)

    return x_history, y_history, theta_history


def plot_simulation(x_history, y_history, theta_history, title, robot_width, robot_length):
    """
    Plots the robot's movement path.

    Args:
        x_history (list): List of x-coordinates.
        y_history (list): List of y-coordinates.
        theta_history (list): List of theta (orientation) values.
        title (str): Title for the plot.
        robot_width (float): Robot width for plotting.
        robot_length (float): Robot length for plotting.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the robot's path
    ax.plot(x_history, y_history, color='cyan', linestyle='-', linewidth=2, label='Robot Path')
    ax.scatter(x_history[0], y_history[0], color='green', s=100, zorder=5, label='Start Point')
    ax.scatter(x_history[-1], y_history[-1], color='red', s=100, zorder=5, label='End Point')

    # Add a marker to show the robot's orientation at the end
    end_x = x_history[-1]
    end_y = y_history[-1]
    end_theta = theta_history[-1]

    # Calculate the arrow's length and position
    arrow_length = max(robot_width, robot_length) * 1.5
    ax.arrow(end_x, end_y,
             arrow_length * math.cos(end_theta),
             arrow_length * math.sin(end_theta),
             head_width=0.2, head_length=0.2, fc='yellow', ec='yellow', zorder=10, label='Final Heading')

    # Configure plot appearance
    ax.set_title(title, color='white')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main simulation examples
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define robot parameters
    WHEEL_RADIUS = 0.1  # meters
    ROBOT_WIDTH = 0.5  # meters
    ROBOT_LENGTH = 0.5  # meters
    DURATION = 10.0  # seconds
    DT = 0.01  # seconds

    # --- Example 1: Straight forward movement ---
    vx_1, vy_1, omega_1 = 0.5, 0.0, 0.0
    x_path_1, y_path_1, theta_path_1 = simulate_mecanum_movement(
        start_x=0, start_y=0, start_theta=0,
        wheel_radius=WHEEL_RADIUS,
        robot_width=ROBOT_WIDTH,
        robot_length=ROBOT_LENGTH,
        desired_v_x=vx_1, desired_v_y=vy_1, desired_omega=omega_1,
        duration=DURATION, dt=DT
    )
    plot_simulation(x_path_1, y_path_1, theta_path_1, f"Straight Forward (vx={vx_1}, vy={vy_1}, omega={omega_1})",
                    ROBOT_WIDTH, ROBOT_LENGTH)

    # --- Example 2: Pure sideways movement (strafing) ---
    vx_2, vy_2, omega_2 = 0.0, 0.5, 0.0
    x_path_2, y_path_2, theta_path_2 = simulate_mecanum_movement(
        start_x=0, start_y=0, start_theta=0,
        wheel_radius=WHEEL_RADIUS,
        robot_width=ROBOT_WIDTH,
        robot_length=ROBOT_LENGTH,
        desired_v_x=vx_2, desired_v_y=vy_2, desired_omega=omega_2,
        duration=DURATION, dt=DT
    )
    plot_simulation(x_path_2, y_path_2, theta_path_2, f"Pure Sideways (vx={vx_2}, vy={vy_2}, omega={omega_2})",
                    ROBOT_WIDTH, ROBOT_LENGTH)

    # --- Example 3: Rotation in place ---
    vx_3, vy_3, omega_3 = 0.0, 0.0, math.radians(30)  # 30 degrees per second
    x_path_3, y_path_3, theta_path_3 = simulate_mecanum_movement(
        start_x=0, start_y=0, start_theta=0,
        wheel_radius=WHEEL_RADIUS,
        robot_width=ROBOT_WIDTH,
        robot_length=ROBOT_LENGTH,
        desired_v_x=vx_3, desired_v_y=vy_3, desired_omega=omega_3,
        duration=DURATION, dt=DT
    )
    plot_simulation(x_path_3, y_path_3, theta_path_3,
                    f"Rotation in Place (vx={vx_3}, vy={vy_3}, omega={math.degrees(omega_3):.2f}°/s)", ROBOT_WIDTH,
                    ROBOT_LENGTH)

    # --- Example 4: Combined forward and sideways movement (diagonal) ---
    vx_4, vy_4, omega_4 = 0.5, 0.5, 0.0
    x_path_4, y_path_4, theta_path_4 = simulate_mecanum_movement(
        start_x=0, start_y=0, start_theta=0,
        wheel_radius=WHEEL_RADIUS,
        robot_width=ROBOT_WIDTH,
        robot_length=ROBOT_LENGTH,
        desired_v_x=vx_4, desired_v_y=vy_4, desired_omega=omega_4,
        duration=DURATION, dt=DT
    )
    plot_simulation(x_path_4, y_path_4, theta_path_4, f"Diagonal Movement (vx={vx_4}, vy={vy_4}, omega={omega_4})",
                    ROBOT_WIDTH, ROBOT_LENGTH)
