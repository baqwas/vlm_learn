import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
GRAVITY = 9.81  # m/s^2


class PerfectHitLauncher3D:
    def __init__(self, target_pos):
        """
        Initialize launcher with a FIXED target position.
        All trajectories will be calculated to HIT this target exactly.
        """
        self.target = np.array(target_pos, dtype=float)
        self.trajectories = []

        if len(self.target) != 3:
            raise ValueError("Target position must be a 3-element array (x, y, z)")

    def solve_for_perfect_hit(self, robot_pos, robot_vel, desired_time=None):
        """
        Solve for launch parameters to hit target exactly.

        Parameters:
        - robot_pos: [x, y, z] starting position
        - robot_vel: [vx, vy, vz] robot velocity
        - desired_time: optional time of flight (if None, solve for minimum time)

        Returns:
        - launch_speed, azimuth_deg, elevation_deg, actual_time_of_flight
        """
        rx, ry, rz = robot_pos
        rvx, rvy, rvz = robot_vel
        tx, ty, tz = self.target

        # If no desired time, solve for minimum time (positive real solution)
        if desired_time is None:
            # Solve for time from z-equation: tz = rz + (rvz + v_launch_z)*t - 0.5*g*t^2
            # => 0.5*g*t^2 - (rvz + v_launch_z)*t + (tz - rz) = 0
            # But we don't know v_launch_z yet — so we need to use x,y to relate

            # Instead, we'll use a different approach: choose a reasonable time
            # and solve for required velocity components

            # Try to find a positive time t such that we can solve for velocity
            # We'll use the horizontal distance to estimate a reasonable time

            horizontal_distance = np.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)
            if horizontal_distance < 0.1:
                # Very close horizontally — use small time
                t = 0.5
            else:
                # Assume average horizontal speed of 10 m/s
                t = horizontal_distance / 10.0

            # Now solve for required launch velocity components
            # From x: tx = rx + (rvx + v_launch_x) * t
            # => v_launch_x = (tx - rx)/t - rvx
            # Similarly for y and z

            v_launch_x = (tx - rx) / t - rvx
            v_launch_y = (ty - ry) / t - rvy
            v_launch_z = (tz - rz + 0.5 * GRAVITY * t ** 2) / t - rvz

            # Calculate launch speed and angles
            launch_speed = np.sqrt(v_launch_x ** 2 + v_launch_y ** 2 + v_launch_z ** 2)

            # Azimuth: angle in XY plane
            azimuth_rad = np.arctan2(v_launch_y, v_launch_x)
            azimuth_deg = np.degrees(azimuth_rad)

            # Elevation: angle above XY plane
            horizontal_speed = np.sqrt(v_launch_x ** 2 + v_launch_y ** 2)
            if horizontal_speed == 0:
                elevation_deg = 90.0 if v_launch_z >= 0 else -90.0
            else:
                elevation_rad = np.arctan2(v_launch_z, horizontal_speed)
                elevation_deg = np.degrees(elevation_rad)

            return launch_speed, azimuth_deg, elevation_deg, t

        else:
            # Use specified time
            t = desired_time
            v_launch_x = (tx - rx) / t - rvx
            v_launch_y = (ty - ry) / t - rvy
            v_launch_z = (tz - rz + 0.5 * GRAVITY * t ** 2) / t - rvz

            launch_speed = np.sqrt(v_launch_x ** 2 + v_launch_y ** 2 + v_launch_z ** 2)

            azimuth_rad = np.arctan2(v_launch_y, v_launch_x)
            azimuth_deg = np.degrees(azimuth_rad)

            horizontal_speed = np.sqrt(v_launch_x ** 2 + v_launch_y ** 2)
            if horizontal_speed == 0:
                elevation_deg = 90.0 if v_launch_z >= 0 else -90.0
            else:
                elevation_rad = np.arctan2(v_launch_z, horizontal_speed)
                elevation_deg = np.degrees(elevation_rad)

            return launch_speed, azimuth_deg, elevation_deg, t

    def compute_trajectory(self, robot_pos, robot_vel, launch_speed, azimuth_deg, elevation_deg, dt=0.01):
        """
        Compute 3D trajectory of ball launched from moving robot.
        """
        # Convert angles to radians
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)

        # Launch velocity components relative to robot
        v_launch_x = launch_speed * np.cos(el_rad) * np.cos(az_rad)
        v_launch_y = launch_speed * np.cos(el_rad) * np.sin(az_rad)
        v_launch_z = launch_speed * np.sin(el_rad)

        # Total initial velocity = robot velocity + launch velocity
        v0_x = robot_vel[0] + v_launch_x
        v0_y = robot_vel[1] + v_launch_y
        v0_z = robot_vel[2] + v_launch_z

        # Initial position
        x0, y0, z0 = robot_pos

        # Time array until ball hits target (or slightly beyond)
        max_time = 15.0
        t = np.arange(0, max_time, dt)

        # Kinematic equations
        x = x0 + v0_x * t
        y = y0 + v0_y * t
        z = z0 + v0_z * t - 0.5 * GRAVITY * t ** 2

        # Find when we're closest to target (should be exact if calculation is correct)
        target_dist = np.sqrt((x - self.target[0]) ** 2 + (y - self.target[1]) ** 2 + (z - self.target[2]) ** 2)
        closest_idx = np.argmin(target_dist)

        # Truncate to show up to closest point (should be target)
        t = t[:closest_idx + 1]
        x = x[:closest_idx + 1]
        y = y[:closest_idx + 1]
        z = z[:closest_idx + 1]

        return t, x, y, z

    def add_perfect_hit_scenario(self, robot_pos, robot_vel, label=""):
        """Add a scenario that guarantees a perfect hit to the fixed target."""
        # Solve for launch parameters to hit target
        launch_speed, azimuth, elevation, time_of_flight = self.solve_for_perfect_hit(robot_pos, robot_vel)

        # Compute trajectory
        traj = self.compute_trajectory(robot_pos, robot_vel, launch_speed, azimuth, elevation)

        # Verify final position is very close to target
        t, x, y, z = traj
        final_pos = np.array([x[-1], y[-1], z[-1]])
        distance_to_target = np.linalg.norm(final_pos - self.target)

        if distance_to_target > 0.01:
            print(f"⚠️ Warning: Final position {final_pos} is {distance_to_target:.4f}m from target.")

        self.trajectories.append({
            'label': label or f"Robot {len(self.trajectories) + 1}",
            'robot_pos': np.array(robot_pos, dtype=float),
            'robot_vel': np.array(robot_vel, dtype=float),
            'launch_params': {
                'speed': float(launch_speed),
                'azimuth': float(azimuth),
                'elevation': float(elevation),
                'time_of_flight': float(time_of_flight)
            },
            'trajectory': traj,
            'distance_to_target': distance_to_target
        })

    def plot_3d(self):
        """Plot all perfect-hit trajectories in 3D."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot FIXED target
        tx, ty, tz = self.target
        ax.scatter(tx, ty, tz, c='red', s=300, marker='*', edgecolors='black', linewidth=3, label='FIXED TARGET')
        ax.text(tx, ty, tz, '  TARGET\n(x=%.1f,y=%.1f,z=%.1f)' % (tx, ty, tz),
                color='red', fontsize=12, fontweight='bold', ha='center')

        # Define colors for different robots
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'gray']

        for i, traj_data in enumerate(self.trajectories):
            t, x, y, z = traj_data['trajectory']
            color = colors[i % len(colors)]
            label = traj_data['label']

            # Plot trajectory
            ax.plot(x, y, z, color=color, linewidth=2.5, label=label)

            # Plot robot start position
            rx, ry, rz = traj_data['robot_pos']
            ax.scatter(rx, ry, rz, c=color, s=100, marker='o', edgecolors='black', linewidth=2)
            ax.text(rx, ry, rz, f' R{i + 1}', color=color, fontsize=10, fontweight='bold', ha='left')

            # Mark final point (should be very close to target)
            ax.scatter(x[-1], y[-1], z[-1], c=color, s=60, marker='x', linewidth=2)

            # Add distance label
            dist = traj_data['distance_to_target']
            ax.text(x[-1], y[-1], z[-1], f' {dist:.3f}m',
                    color=color, fontsize=8, ha='right')

        # Customize plot
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold')
        ax.set_title('PERFECT-HIT 3D Ball Trajectories to FIXED Target', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set equal scaling
        all_x = [traj_data['trajectory'][1] for traj_data in self.trajectories]
        all_y = [traj_data['trajectory'][2] for traj_data in self.trajectories]
        all_z = [traj_data['trajectory'][3] for traj_data in self.trajectories]

        bounds = [
            min(np.min(arr) for arr in all_x + [[tx]]),
            max(np.max(arr) for arr in all_x + [[tx]]),
            min(np.min(arr) for arr in all_y + [[ty]]),
            max(np.max(arr) for arr in all_y + [[ty]]),
            min(np.min(arr) for arr in all_z + [[tz]]),
            max(np.max(arr) for arr in all_z + [[tz]])
        ]

        mid_x = (bounds[0] + bounds[1]) / 2
        mid_y = (bounds[2] + bounds[3]) / 2
        mid_z = (bounds[4] + bounds[5]) / 2
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

        ax.set_xlim(mid_x - span / 2, mid_x + span / 2)
        ax.set_ylim(mid_y - span / 2, mid_y + span / 2)
        ax.set_zlim(mid_z - span / 2, mid_z + span / 2)

        plt.tight_layout()
        plt.show()


# =============================
# MAIN DEMONSTRATION - PERFECT HITS TO FIXED TARGET
# =============================

if __name__ == "__main__":
    # 🔒 FIXED TARGET POSITION — NEVER CHANGES
    FIXED_TARGET = (10.0, 8.0, 2.0)  # X, Y, Z — THIS IS THE ONLY TARGET FOR ALL SHOTS

    print("🚀 3D BALL LAUNCH SIMULATION WITH PERFECT HITS TO FIXED TARGET")
    print(f"🎯 Target is FIXED at: ({FIXED_TARGET[0]:.1f}, {FIXED_TARGET[1]:.1f}, {FIXED_TARGET[2]:.1f})")
    print("-" * 70)

    launcher = PerfectHitLauncher3D(FIXED_TARGET)

    # Add multiple robot launch scenarios — ALL WILL HIT THE TARGET EXACTLY
    scenarios = [
        {
            'robot_pos': (0.0, 0.0, 0.0),
            'robot_vel': (1.0, 0.5, 0.0),
            'label': 'Robot A: Near Origin'
        },
        {
            'robot_pos': (-3.0, 5.0, 1.0),
            'robot_vel': (0.0, 0.0, 0.5),
            'label': 'Robot B: Left & Up'
        },
        {
            'robot_pos': (7.0, 3.0, 0.0),
            'robot_vel': (2.0, 1.0, 0.0),
            'label': 'Robot C: Behind Target'
        },
        {
            'robot_pos': (5.0, 10.0, 4.0),
            'robot_vel': (0.0, -1.0, -0.5),
            'label': 'Robot D: Above & Back'
        },
        {
            'robot_pos': (-8.0, -2.0, 0.0),
            'robot_vel': (3.0, 2.0, 0.0),
            'label': 'Robot E: Far Away'
        }
    ]

    for scenario in scenarios:
        launcher.add_perfect_hit_scenario(
            robot_pos=scenario['robot_pos'],
            robot_vel=scenario['robot_vel'],
            label=scenario['label']
        )

    # Print summary showing perfect hits
    print("📌 All shots are calculated to hit the SAME FIXED TARGET exactly.")
    print("✅ Distance shown is from final ball position to FIXED TARGET (should be near zero).\n")

    for i, traj in enumerate(launcher.trajectories):
        t, x, y, z = traj['trajectory']
        final_pos = np.array([x[-1], y[-1], z[-1]])
        distance_to_fixed_target = traj['distance_to_target']

        print(f"{traj['label']}:")
        print(f"  Robot Start: ({traj['robot_pos'][0]:.1f}, {traj['robot_pos'][1]:.1f}, {traj['robot_pos'][2]:.1f})")
        print(
            f"  Launch: {traj['launch_params']['speed']:.2f} m/s @ {traj['launch_params']['azimuth']:.1f}° azimuth, {traj['launch_params']['elevation']:.1f}° elevation")
        print(f"  Time of Flight: {traj['launch_params']['time_of_flight']:.2f} s")
        print(f"  Final Pos: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
        print(f"  Distance to FIXED Target: {distance_to_fixed_target:.6f} m")
        print()

    # Generate 3D plot with PERFECT HIT trajectories
    launcher.plot_3d()

