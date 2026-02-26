import matplotlib.pyplot as plt
import numpy as np

# 1. Define the simulation data for the environment map
# We use simple (x, y) coordinates for key objects and the agent
object_positions = {
    "Red Box": (10, 5),
    "Blue Cylinder": (20, 15),
    "Green Sphere": (11, 4)
}
agent_position_turn_1 = (5, 10)  # Starting position

# 2. Create the plot for Turn 1 (Initial Map)
fig, ax = plt.subplots(figsize=(8, 8))

# Set limits and maintain aspect ratio
ax.set_xlim(0, 25)
ax.set_ylim(0, 20)
ax.set_aspect('equal', adjustable='box')

# Plot objects
for name, (x, y) in object_positions.items():
    if name == "Red Box":
        ax.plot(x, y, 'rs', markersize=15, label=name)  # red square
    elif name == "Blue Cylinder":
        ax.plot(x, y, 'bo', markersize=15, label=name)  # blue circle (cylinder top)
    elif name == "Green Sphere":
        ax.plot(x, y, 'gD', markersize=10, label=name)  # green diamond (sphere)

# Plot Agent (Turn 1)
ax.plot(agent_position_turn_1[0], agent_position_turn_1[1], 'k^', markersize=12, label='Agent Start Position')
ax.annotate('Agent Start', agent_position_turn_1, textcoords="offset points", xytext=(5, -10), ha='center', fontsize=9)

# Draw a simulated bounding box to represent perception/grounding (Green Sphere next to Red Box)
ax.plot([10, 11, 11, 10, 10], [4, 4, 5, 5, 4], 'k--', alpha=0.5, linewidth=1)
ax.text(10.5, 3.5, 'Crucial Detail Area', fontsize=8, ha='center', color='gray')

# Annotations for coordinates (essential for the reasoning trace)
for name, (x, y) in object_positions.items():
    ax.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

# Set labels and title
ax.set_title('Simulated Environment Map (Turn 1: Initial Observation)', fontsize=14)
ax.set_xlabel('X Coordinate (Units)', fontsize=10)
ax.set_ylabel('Y Coordinate (Units)', fontsize=10)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, linestyle=':', alpha=0.5)

# Save the plot
plt.savefig('initial_environment_map.png')