# -----------------------------------------------------------------------------
# mecanum_robot_astar_path_planning_improved.py
#
# A Python script to simulate a four-wheel mecanum drive robot finding and
# following a path around obstacles using the A* pathfinding algorithm. The
# visualization has been improved to clearly show obstacle avoidance.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import math
import heapq


# --- A* Pathfinding Algorithm Implementation ---

class Node:
    """A node class for A* Pathfinding."""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0  # Cost from start node
        self.h = 0  # Heuristic cost to end node
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


def a_star_search(grid, start, end):
    """
    Finds the shortest path from start to end on a grid with obstacles using A*.

    Args:
        grid (np.ndarray): A 2D numpy array representing the environment.
                           0 = traversable, 1 = obstacle.
        start (tuple): The (row, col) coordinates of the starting point.
        end (tuple): The (row, col) coordinates of the end point.

    Returns:
        list: A list of (x, y) coordinates representing the path, or None if no path is found.
    """
    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_list = set()
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if current_node.position == end_node.position:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for position in neighbors:
            node_position = (current_node.position[0] + position[0], current_node.position[1] + position[1])

            if (node_position[0] < 0 or node_position[0] >= grid.shape[0] or
                    node_position[1] < 0 or node_position[1] >= grid.shape[1]):
                continue

            if grid[node_position[0], node_position[1]] == 1:
                continue

            if node_position in closed_list:
                continue

            new_node = Node(current_node, node_position)

            new_node.g = current_node.g + math.sqrt(position[0] ** 2 + position[1] ** 2)
            new_node.h = math.sqrt((new_node.position[0] - end_node.position[0]) ** 2 +
                                   (new_node.position[1] - end_node.position[1]) ** 2)
            new_node.f = new_node.g + new_node.h

            in_open_list = any(node.position == new_node.position and node.g <= new_node.g for node in open_list)
            if in_open_list:
                continue

            heapq.heappush(open_list, new_node)

    return None


# --- Mecanum Robot Simulation ---

def follow_path_from_astar(start_x, start_y, start_theta, robot_width, robot_length, waypoints, k_p_linear, k_p_angular,
                           duration, dt, grid_scale):
    """
    Simulates a mecanum robot following a series of waypoints from an A* path.
    """
    x_history = [start_x]
    y_history = [start_y]
    theta_history = [start_theta]
    current_x, current_y, current_theta = start_x, start_y, start_theta
    waypoint_index = 0
    waypoint_threshold = grid_scale * 1.5

    for _ in np.arange(0, duration, dt):
        if waypoint_index >= len(waypoints):
            desired_v_x = 0
            desired_v_y = 0
            desired_omega = 0
        else:
            target_x_grid, target_y_grid = waypoints[waypoint_index]
            target_x = target_y_grid * grid_scale
            target_y = target_x_grid * grid_scale

            error_x = target_x - current_x
            error_y = target_y - current_y
            distance_to_target = math.sqrt(error_x ** 2 + error_y ** 2)

            if distance_to_target < waypoint_threshold:
                print(f"Waypoint {waypoint_index + 1} reached.")
                waypoint_index += 1
                continue

            desired_v_forward = k_p_linear * distance_to_target
            desired_theta = math.atan2(error_y, error_x)
            heading_error = desired_theta - current_theta
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            desired_omega = k_p_angular * heading_error

            desired_v_x = desired_v_forward * math.cos(heading_error)
            desired_v_y = desired_v_forward * math.sin(heading_error)

        d_x = (desired_v_x * math.cos(current_theta) - desired_v_y * math.sin(current_theta)) * dt
        d_y = (desired_v_x * math.sin(current_theta) + desired_v_y * math.cos(current_theta)) * dt
        d_theta = desired_omega * dt

        current_x += d_x
        current_y += d_y
        current_theta += d_theta

        x_history.append(current_x)
        y_history.append(current_y)
        theta_history.append(current_theta)

    return x_history, y_history, theta_history


def plot_simulation_with_grid(x_history, y_history, theta_history, waypoints, grid, grid_scale, title):
    """
    Plots the robot's movement path, A* path, and the grid with obstacles.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the grid and obstacles
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 1:
                # Fill the obstacle cells with a distinct color (blue)
                ax.fill([c * grid_scale, (c + 1) * grid_scale, (c + 1) * grid_scale, c * grid_scale],
                        [r * grid_scale, r * grid_scale, (r + 1) * grid_scale, (r + 1) * grid_scale],
                        color='blue', alpha=0.8, edgecolor='black', linewidth=1.5, zorder=1)

    # Plot the robot's actual path
    ax.plot(x_history, y_history, color='cyan', linestyle='-', linewidth=2, label='Robot Path')
    ax.scatter(x_history[0], y_history[0], color='green', s=100, zorder=5, label='Start Point')
    ax.scatter(x_history[-1], y_history[-1], color='red', s=100, zorder=5, label='End Point')

    # Plot the A* planned path
    if waypoints:
        waypoints_x = [wp[1] * grid_scale for wp in waypoints]
        waypoints_y = [wp[0] * grid_scale for wp in waypoints]
        ax.plot(waypoints_x, waypoints_y, color='yellow', linestyle='--', alpha=0.5, label='A* Planned Path')
        ax.scatter(waypoints_x, waypoints_y, color='yellow', marker='s', s=40, zorder=4, label='A* Waypoints')

    ax.set_title(title, color='white')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main simulation example with A* pathfinding
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define robot and simulation parameters
    ROBOT_WIDTH = 0.5
    ROBOT_LENGTH = 0.5
    DURATION = 60.0
    DT = 0.01
    GRID_SCALE = 0.5  # Each grid cell represents 0.5m x 0.5m

    # Proportional control gains
    K_P_LINEAR = 0.5
    K_P_ANGULAR = 1.0

    # Define the environment grid and obstacles
    GRID_SIZE = 20
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    # Add more complex obstacles to the grid
    grid[2:10, 5] = 1
    grid[10, 5:15] = 1
    grid[10:18, 15] = 1
    grid[12:18, 2] = 1
    grid[8, 10:13] = 1

    # Define start and end points in grid coordinates
    start_grid = (1, 1)
    end_grid = (18, 18)

    # Check if start and end points are in a clear space
    if grid[start_grid] == 1 or grid[end_grid] == 1:
        print("Start or end point is inside an obstacle. Please choose a new location.")
    else:
        # Step 1: Find the path using A*
        astar_path = a_star_search(grid, start_grid, end_grid)

        if astar_path:
            # Step 2: Simulate the robot following the A* path
            x_path, y_path, theta_path = follow_path_from_astar(
                start_x=start_grid[1] * GRID_SCALE,
                start_y=start_grid[0] * GRID_SCALE,
                start_theta=0,
                robot_width=ROBOT_WIDTH,
                robot_length=ROBOT_LENGTH,
                waypoints=astar_path,
                k_p_linear=K_P_LINEAR,
                k_p_angular=K_P_ANGULAR,
                duration=DURATION,
                dt=DT,
                grid_scale=GRID_SCALE
            )
            plot_simulation_with_grid(x_path, y_path, theta_path, astar_path, grid, GRID_SCALE,
                                      "Mecanum Robot A* Pathfinding")
        else:
            print("No path found between the start and end points.")
