#!/usr/bin/env python3
"""
offset_diagram.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Constants from the Java program
CAMERA_OFFSET_INCHES = 3.5  # Horizontal offset of the camera from the robot's center
DESIRED_DISTANCE = 12.0  # Desired distance from the robot's center to the tag


def create_diagram(offset):
    """
    Generates a visual diagram of the robot, camera, and AprilTag.

    Args:
        offset (float): The horizontal distance of the camera from the robot's center.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal", adjustable="box")
    plt.style.use("dark_background")

    # Set plot limits
    ax.set_xlim(-15, 15)
    ax.set_ylim(-5, 25)

    # Draw the AprilTag
    tag_rect = patches.Rectangle(
        (-3, DESIRED_DISTANCE + 10),
        2,
        2,
        linewidth=2,
        edgecolor="cyan",
        facecolor="aqua",
        alpha=0.6,
    )
    ax.add_patch(tag_rect)
    ax.text(
        -2.15,
        DESIRED_DISTANCE + 9,
        "AprilTag",
        ha="center",
        va="center",
        color="black",
        weight="bold",
    )

    # Draw the Robot
    robot_width = 13
    robot_height = 13
    robot_center_x = 0
    robot_center_y = 0
    robot_rect = patches.Rectangle(
        (robot_center_x - robot_width / 2, robot_center_y),
        robot_width,
        robot_height,
        linewidth=2,
        edgecolor="lime",
        facecolor="green",
        alpha=0.4,
    )
    ax.add_patch(robot_rect)
    ax.text(
        robot_center_x,
        robot_center_y + robot_height / 2,
        "Robot Body",
        ha="center",
        va="center",
        color="white",
        weight="bold",
    )

    # Draw the Camera
    camera_size = 3
    camera_x = offset
    camera_y = robot_center_y + robot_height
    camera_rect = patches.Rectangle(
        (camera_x - camera_size / 2, camera_y - camera_size / 2),
        camera_size,
        camera_size,
        linewidth=2,
        edgecolor="red",
        facecolor="red",
        alpha=0.8,
    )
    ax.add_patch(camera_rect)
    ax.text(
        camera_x,
        camera_y,
        "Camera",
        ha="center",
        va="bottom",
        color="white",
        weight="bold",
    )

    # Draw lines and labels to illustrate the offset and distance
    # Line for desired distance
    ax.plot([0, 0], [robot_center_y, DESIRED_DISTANCE], "w--", lw=1)
    ax.text(
        0.5,
        DESIRED_DISTANCE / 2,
        f'Desired Distance\n({DESIRED_DISTANCE}" from robot center)',
        ha="left",
        va="center",
        color="white",
    )

    # Line for camera offset
    ax.plot([robot_center_x, camera_x], [camera_y, camera_y], "y--", lw=1)
    ax.text(
        camera_x / 2,
        camera_y + 3,
        f'Camera Offset ({offset}")',
        ha="center",
        va="bottom",
        color="pink",
    )

    # Line showing final desired alignment
    ax.plot([0, 0], [DESIRED_DISTANCE, DESIRED_DISTANCE + 5], "w-", lw=1)
    ax.text(
        0,
        DESIRED_DISTANCE + 5.5,
        "Final Robot Position",
        ha="center",
        va="bottom",
        color="white",
        weight="bold",
    )

    ax.set_title("Robot-AprilTag Alignment with Camera Offset Correction")
    ax.set_xlabel("Horizontal Position (Inches)")
    ax.set_ylabel("Forward Position (Inches)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend([robot_rect, camera_rect, tag_rect], ["Robot", "Camera", "AprilTag"])

    plt.show()


if __name__ == "__main__":
    create_diagram(CAMERA_OFFSET_INCHES)
