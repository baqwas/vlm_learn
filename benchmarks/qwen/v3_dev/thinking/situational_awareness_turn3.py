#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
VERSION: 1.0.1
UPDATED: 2026-03-05 08:13:21
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved.
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: [REPLACE WITH FILE DESCRIPTION]
================================================================================
"""
"""
situational_awareness_turn3.py
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
# Agent must return to the initial starting position for the final action
agent_position_turn_3 = (5, 10)

# 2. Create the plot for Turn 3 (Memory Retrieval and Action)
fig, ax = plt.subplots(figsize=(8, 8))

# Set limits and aspect ratio
ax.set_xlim(0, 25)
ax.set_ylim(0, 20)
ax.set_aspect("equal", adjustable="box")

# Plot objects
for name, (x, y) in object_positions.items():
    if name == "Red Box":
        ax.plot(x, y, "rs", markersize=15, label=name)
    elif name == "Blue Cylinder":
        # Faded as it is not the focus of this memory-driven action
        ax.plot(x, y, "bo", markersize=15, label=name, alpha=0.5)
    elif name == "Green Sphere":
        ax.plot(x, y, "gD", markersize=15, label=name)  # Highlighted as the target

# Plot Agent at RETURNED Position
ax.plot(
    agent_position_turn_3[0],
    agent_position_turn_3[1],
    "k^",
    markersize=12,
    label="Agent Current Position (Returned to Start)",
)
ax.annotate(
    "Agent Returned to Start",
    agent_position_turn_3,
    textcoords="offset points",
    xytext=(-20, 5),
    ha="right",
    fontsize=9,
    fontweight="bold",
)

# Highlight the Memory Retrieval and Action Focus (The Green Sphere next to the Red Box)
# Re-draw the bounding box from Turn 1 to show memory recall
ax.plot(
    [10, 11, 11, 10, 10],
    [4, 4, 5, 5, 4],
    "k--",
    alpha=0.8,
    linewidth=2,
    label="Memory Focus Area (Turn 1)",
)
ax.text(
    10.5,
    3.5,
    "Crucial Detail Recalled (Green Sphere)",
    fontsize=9,
    ha="center",
    color="black",
    fontweight="bold",
)

# Visualize the final action target
ax.annotate(
    "TARGET: Pick up Green Sphere",
    object_positions["Green Sphere"],
    textcoords="offset points",
    xytext=(15, 0),
    arrowprops=dict(arrowstyle="->", color="g", linewidth=2),
    fontsize=10,
    fontweight="bold",
)

# Annotations for coordinates
for name, (x, y) in object_positions.items():
    ax.annotate(
        f"({x}, {y})", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8
    )

# Set labels and title
ax.set_title(
    "Simulated Environment Map (Turn 3: Long-Context Memory & Action)", fontsize=14
)
ax.set_xlabel("X Coordinate (Units)", fontsize=10)
ax.set_ylabel("Y Coordinate (Units)", fontsize=10)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.5)

# Save the plot
plt.savefig("turn_3_environment_map.png")
