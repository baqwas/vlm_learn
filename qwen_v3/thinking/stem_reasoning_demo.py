import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 1. Define the coordinates
# Square: s=5, vertices (0,0), (5,0), (5,5), (0,5)
square_coords = [(0, 0), (5, 0), (5, 5), (0, 5)]

# Triangle: base b=5 (shared with right side of square), height h=2
# Shared side: (5,0) to (5,5). Third vertex at x=7 (5+2), y=2.5 (midpoint for visual clarity)
triangle_coords = [(5, 0), (5, 5), (7, 2.5)]

# 2. Setup the plot
fig, ax = plt.subplots(figsize=(8, 6))

# 3. Draw the shapes
# Draw Square
square = Polygon(square_coords, closed=True, edgecolor='black', facecolor='lightblue', linewidth=2)
ax.add_patch(square)

# Draw Triangle
triangle = Polygon(triangle_coords, closed=True, edgecolor='black', facecolor='lightcoral', linewidth=2)
ax.add_patch(triangle)

# 4. Add dimension labels
# Label for Square side (bottom)
ax.text(2.5, -0.5, '5', ha='center', va='top', fontsize=14, fontweight='bold')
# Label for Square side (left)
ax.text(-0.5, 2.5, '5', ha='right', va='center', fontsize=14, fontweight='bold')

# Label for Triangle height (horizontal distance from shared side)
# Draw a dashed line to indicate the height h=2
ax.plot([5, 7], [2.5, 2.5], 'k--', linewidth=1)
ax.text(6, 3, 'h = 2', ha='center', va='bottom', fontsize=14, color='darkred', fontweight='bold')

# Label for Triangle base (shared side)
ax.text(5.5, 2.5, 'b = 5', ha='left', va='center', fontsize=14, color='darkgreen', fontweight='bold', rotation=90)


# 5. Set plot properties
ax.set_xlim(-1, 8)
ax.set_ylim(-1, 6)
ax.set_aspect('equal', adjustable='box')
ax.axis('off') # Hide axes

# Add a title
plt.title('Composite Figure for Area Calculation', fontsize=16)

# 6. Save the figure
figure_filename = '../images/composite_area_figure.png'
plt.savefig(figure_filename, bbox_inches='tight')