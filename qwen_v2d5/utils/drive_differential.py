# -----------------------------------------------------------------------------
# two_wheel_robot_sim.py
#
# A Python script to simulate the movement of a two-wheel differential drive
# robot on a 2D plane from point A to point B.
#
# The simulation uses matplotlib for visualization and includes detailed
# comments to explain the underlying kinematics.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import math

def simulate_robot_movement(start_x, start_y, start_theta, wheel_base, left_wheel_speed, right_wheel_speed, duration, dt):
    """
    Simulates the movement of a two-wheel differential drive robot.

    Args:
        start_x (float): Initial x-coordinate of the robot.
        start_y (float): Initial y-coordinate of the robot.
        start_theta (float): Initial orientation of the robot in radians.
        wheel_base (float): Distance between the two wheels.
        left_wheel_speed (float): Speed of the left wheel.
        right_wheel_speed (float): Speed of the right wheel.
        duration (float): Total simulation time in seconds.
        dt (float): Time step for the simulation.

    Returns:
        tuple: A tuple containing lists of x, y, and theta values over time.
    """
    # Initialize lists to store the robot's path and orientation over time
    x_history = [start_x]
    y_history = [start_y]
    theta_history = [start_theta]

    # Initialize current position and orientation
    current_x = start_x
    current_y = start_y
    current_theta = start_theta

    # Calculate the linear and angular velocity of the robot based on wheel speeds
    # Linear velocity (v) is the average of the two wheel speeds.
    # Angular velocity (omega) is the difference in wheel speeds divided by the wheel base.
    v = (right_wheel_speed + left_wheel_speed) / 2.0
    omega = (right_wheel_speed - left_wheel_speed) / wheel_base

    # The simulation loop iterates over the specified duration
    for _ in range(int(duration / dt)):
        # Calculate the change in orientation and position for this time step
        d_theta = omega * dt
        d_x = v * math.cos(current_theta) * dt
        d_y = v * math.sin(current_theta) * dt

        # Update the robot's position and orientation
        current_x += d_x
        current_y += d_y
        current_theta += d_theta

        # Store the new state in the history lists
        x_history.append(current_x)
        y_history.append(current_y)
        theta_history.append(current_theta)

    return x_history, y_history, theta_history

def plot_simulation(x_history, y_history, theta_history, title, wheel_base):
    """
    Plots the robot's movement path.

    Args:
        x_history (list): List of x-coordinates.
        y_history (list): List of y-coordinates.
        theta_history (list): List of theta (orientation) values.
        title (str): Title for the plot.
        wheel_base (float): Distance between the two wheels.
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
    arrow_length = wheel_base * 1.5
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
    WHEEL_BASE = 1.0  # meters
    DURATION = 10.0   # seconds
    DT = 0.01         # seconds

    # --- Example 1: Straight line movement ---
    # Both wheels move at the same speed.
    left_speed_1 = 1.0  # m/s
    right_speed_1 = 1.0 # m/s
    x_path_1, y_path_1, theta_path_1 = simulate_robot_movement(
        start_x=0, start_y=0, start_theta=0,
        wheel_base=WHEEL_BASE,
        left_wheel_speed=left_speed_1,
        right_wheel_speed=right_speed_1,
        duration=DURATION,
        dt=DT
    )
    plot_simulation(x_path_1, y_path_1, theta_path_1, "Straight Line Movement (v_L=1, v_R=1)", WHEEL_BASE)

    # --- Example 2: Turning (Spin in place) ---
    # Wheels move at the same speed in opposite directions.
    left_speed_2 = -0.5 # m/s
    right_speed_2 = 0.5  # m/s
    x_path_2, y_path_2, theta_path_2 = simulate_robot_movement(
        start_x=0, start_y=0, start_theta=math.radians(90),
        wheel_base=WHEEL_BASE,
        left_wheel_speed=left_speed_2,
        right_wheel_speed=right_speed_2,
        duration=DURATION,
        dt=DT
    )
    plot_simulation(x_path_2, y_path_2, theta_path_2, "Turning in Place (v_L=-0.5, v_R=0.5)", WHEEL_BASE)

    # --- Example 3: Arc Movement ---
    # One wheel moves faster than the other.
    left_speed_3 = 0.5  # m/s
    right_speed_3 = 1.0 # m/s
    x_path_3, y_path_3, theta_path_3 = simulate_robot_movement(
        start_x=0, start_y=0, start_theta=0,
        wheel_base=WHEEL_BASE,
        left_wheel_speed=left_speed_3,
        right_wheel_speed=right_speed_3,
        duration=DURATION,
        dt=DT
    )
    plot_simulation(x_path_3, y_path_3, theta_path_3, "Arc Movement (v_L=0.5, v_R=1.0)", WHEEL_BASE)
