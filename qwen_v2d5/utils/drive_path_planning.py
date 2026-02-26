# -----------------------------------------------------------------------------
# mecanum_robot_path_planning.py
#
# A Python script to simulate a four-wheel mecanum drive robot following a
# predefined path of waypoints using a proportional controller.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import math


def follow_path(start_x, start_y, start_theta, wheel_radius, robot_width, robot_length, waypoints, k_p_linear,
                k_p_angular, duration, dt):
    """
    Simulates a mecanum robot following a series of waypoints.

    Args:
        start_x (float): Initial x-coordinate of the robot.
        start_y (float): Initial y-coordinate of the robot.
        start_theta (float): Initial orientation of the robot in radians.
        wheel_radius (float): Radius of the mecanum wheels.
        robot_width (float): The width of the robot from wheel to wheel.
        robot_length (float): The length of the robot from wheel to wheel.
        waypoints (list): A list of (x, y) tuples representing the waypoints.
        k_p_linear (float): Proportional gain for linear velocity.
        k_p_angular (float): Proportional gain for angular velocity.
        duration (float): Total simulation time in seconds.
        dt (float): Time step for the simulation.

    Returns:
        tuple: Lists of x, y, and theta values over time.
    """
    # Initialize lists to store the robot's path
    x_history = [start_x]
    y_history = [start_y]
    theta_history = [start_theta]

    # Initialize current position and orientation
    current_x = start_x
    current_y = start_y
    current_theta = start_theta
    waypoint_index = 0
    waypoint_threshold = 0.1  # Distance to consider a waypoint reached

    # The simulation loop iterates over the specified duration
    for _ in np.arange(0, duration, dt):
        if waypoint_index >= len(waypoints):
            # The robot has reached all waypoints, stop moving
            desired_v_x = 0
            desired_v_y = 0
            desired_omega = 0
        else:
            # Get the current target waypoint
            target_x, target_y = waypoints[waypoint_index]

            # Calculate the error (distance and angle to the target)
            error_x = target_x - current_x
            error_y = target_y - current_y
            distance_to_target = math.sqrt(error_x ** 2 + error_y ** 2)

            # Check if the waypoint is reached
            if distance_to_target < waypoint_threshold:
                print(f"Waypoint {waypoint_index + 1} reached.")
                waypoint_index += 1
                continue

            # Calculate the desired linear velocity components
            # This is a proportional control, velocity is proportional to the distance
            desired_v_forward = k_p_linear * distance_to_target

            # Calculate the desired heading towards the target
            desired_theta = math.atan2(error_y, error_x)

            # Calculate the heading error (difference between current and desired orientation)
            heading_error = desired_theta - current_theta

            # Normalize the heading error to be within the range [-pi, pi]
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

            # Calculate the desired angular velocity (proportional to the heading error)
            desired_omega = k_p_angular * heading_error

            # Decompose forward velocity into local x and y components
            desired_v_x = desired_v_forward * math.cos(heading_error)
            desired_v_y = desired_v_forward * math.sin(heading_error)

        # Update the position and orientation of the robot based on the calculated velocities
        d_x = (desired_v_x * math.cos(current_theta) - desired_v_y * math.sin(current_theta)) * dt
        d_y = (desired_v_x * math.sin(current_theta) + desired_v_y * math.cos(current_theta)) * dt
        d_theta = desired_omega * dt

        current_x += d_x
        current_y += d_y
        current_theta += d_theta

        # Store the new state in the history lists
        x_history.append(current_x)
        y_history.append(current_y)
        theta_history.append(current_theta)

    return x_history, y_history, theta_history


def plot_simulation(x_history, y_history, theta_history, waypoints, title, robot_width, robot_length):
    """
    Plots the robot's movement path and the waypoints.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the robot's path
    ax.plot(x_history, y_history, color='cyan', linestyle='-', linewidth=2, label='Robot Path')
    ax.scatter(x_history[0], y_history[0], color='green', s=100, zorder=5, label='Start Point')
    ax.scatter(x_history[-1], y_history[-1], color='red', s=100, zorder=5, label='End Point')

    # Plot the waypoints
    waypoints_x = [wp[0] for wp in waypoints]
    waypoints_y = [wp[1] for wp in waypoints]
    ax.scatter(waypoints_x, waypoints_y, color='yellow', marker='s', s=80, zorder=4, label='Waypoints')

    # Draw lines connecting the waypoints to show the planned path
    ax.plot(waypoints_x, waypoints_y, color='yellow', linestyle='--', alpha=0.5, label='Planned Path')

    # Add a marker to show the robot's orientation at the end
    end_x = x_history[-1]
    end_y = y_history[-1]
    end_theta = theta_history[-1]
    arrow_length = max(robot_width, robot_length) * 1.5
    ax.arrow(end_x, end_y,
             arrow_length * math.cos(end_theta),
             arrow_length * math.sin(end_theta),
             head_width=0.2, head_length=0.2, fc='lime', ec='lime', zorder=10, label='Final Heading')

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
# Main simulation example with waypoints
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define robot and simulation parameters
    WHEEL_RADIUS = 0.1
    ROBOT_WIDTH = 0.5
    ROBOT_LENGTH = 0.5
    DURATION = 60.0  # Increase duration for longer paths
    DT = 0.01

    # Define proportional control gains
    K_P_LINEAR = 0.5  # Controls how quickly the robot moves forward
    K_P_ANGULAR = 1.0  # Controls how quickly the robot turns to face the waypoint

    # Define the waypoints for the robot to follow
    # Start at (0, 0), move to (5, 5), then to (-5, 5), then to (-5, -5), and finally to (0, 0)
    WAYPOINTS = [
        (5.0, 5.0),
        (-5.0, 5.0),
        (-5.0, -5.0),
        (0.0, 0.0)
    ]

    # Run the path planning simulation
    x_path, y_path, theta_path = follow_path(
        start_x=0,
        start_y=0,
        start_theta=0,
        wheel_radius=WHEEL_RADIUS,
        robot_width=ROBOT_WIDTH,
        robot_length=ROBOT_LENGTH,
        waypoints=WAYPOINTS,
        k_p_linear=K_P_LINEAR,
        k_p_angular=K_P_ANGULAR,
        duration=DURATION,
        dt=DT
    )
    plot_simulation(x_path, y_path, theta_path, WAYPOINTS, "Mecanum Robot Path Following", ROBOT_WIDTH, ROBOT_LENGTH)
