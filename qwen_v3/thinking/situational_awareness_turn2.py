#!/usr/bin/env python3
"""
sitational awareness turn2.py
"""
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the simulation data
# Object positions remain CONSTANT (Absolute positions)
object_positions = {
    "Red Box": (10, 5),
    "Blue Cylinder": (20, 15),
    "Green Sphere": (11, 4),
}
# Agent movement: Start (5, 10) -> Move 2 steps left -> New Position (3, 10)
agent_start_position = (5, 10)
agent_new_position = (3, 10)

# 2. Create the plot for Turn 2 (Viewpoint Change)
fig, ax = plt.subplots(figsize=(8, 8))

# Set limits and aspect ratio
ax.set_xlim(0, 25)
ax.set_ylim(0, 20)
ax.set_aspect("equal", adjustable="box")

# Plot objects (same as Turn 1)
for name, (x, y) in object_positions.items():
    if name == "Red Box":
        ax.plot(x, y, "rs", markersize=15, label=name)  # red square
    elif name == "Blue Cylinder":
        ax.plot(x, y, "bo", markersize=15, label=name)  # blue circle (cylinder top)
    elif name == "Green Sphere":
        ax.plot(x, y, "gD", markersize=10, label=name)  # green diamond (sphere)

# Plot Agent at NEW Position
ax.plot(
    agent_new_position[0],
    agent_new_position[1],
    "k^",
    markersize=12,
    label="Agent Current Position",
)
ax.annotate(
    "Agent New Position (3, 10)",
    agent_new_position,
    textcoords="offset points",
    xytext=(-50, 5),
    ha="left",
    fontsize=9,
    fontweight="bold",
)

# Visualize the movement (The "two steps left" instruction)
ax.annotate(
    "",
    xy=agent_new_position,
    xytext=agent_start_position,
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0.0", color="purple", linewidth=2
    ),
    fontsize=8,
)
ax.text(4, 10.5, "2 Steps Left", color="purple", fontsize=9, ha="center")

# Highlight the relative position calculation focus (Red Box to Blue Cylinder)
ax.plot([10, 20], [5, 15], "b:", alpha=0.5, linewidth=1)
ax.text(15, 11, "Relative Position Focus", fontsize=9, ha="center", color="blue")


# Annotations for coordinates
for name, (x, y) in object_positions.items():
    ax.annotate(
        f"({x}, {y})", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8
    )

# Set labels and title
ax.set_title("Simulated Environment Map (Turn 2: Viewpoint Change)", fontsize=14)
ax.set_xlabel("X Coordinate (Units)", fontsize=10)
ax.set_ylabel("Y Coordinate (Units)", fontsize=10)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.5)

# Save the plot
plt.savefig("turn_2_environment_map.png")
