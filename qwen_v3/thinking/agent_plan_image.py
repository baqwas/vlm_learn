import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off') # Hide axes

# --- Visual Elements ---

# 1. User Input (Left) - Represents the multimodal query
ax.add_patch(plt.Rectangle((0.5, 7.5), 2.5, 1.5, color='#ADD8E6', alpha=0.8, edgecolor='black', linewidth=1, zorder=2))
ax.text(1.75, 8.5, "User Input", ha='center', va='center', fontsize=14, fontweight='bold', color='black')
ax.text(1.75, 8.1, "(Image + Text Goal)", ha='center', va='center', fontsize=10, color='black')

# Arrow from User Input to Qwen3-VL
ax.annotate("", xy=(3, 8.25), xytext=(3.7, 8.25), arrowprops=dict(arrowstyle="->", color="gray", lw=2))

# 2. Qwen3-VL Thinking Core (Center)
ax.add_patch(plt.Rectangle((3.7, 1.5), 2.6, 7.5, color='#90EE90', alpha=0.8, edgecolor='black', linewidth=1.5, zorder=2))
ax.text(5, 8.5, "Qwen3-VL", ha='center', va='center', fontsize=16, fontweight='bold', color='darkgreen')
ax.text(5, 8.1, "Thinking Model", ha='center', va='center', fontsize=12, color='darkgreen')

# --- Internal Thinking Steps ---
# Step 1: Perception & Grounding
ax.add_patch(plt.Rectangle((4, 6.5), 2, 0.8, color='white', alpha=0.9, edgecolor='darkgray', linewidth=0.5))
ax.text(5, 6.9, "1. PERCEPTION", ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, 6.7, "(Image Analysis)", ha='center', va='center', fontsize=8)
ax.annotate("", xy=(5, 6.5), xytext=(5, 5.8), arrowprops=dict(arrowstyle="->", color="darkgray", lw=1))

# Step 2: Reasoning & Integration
ax.add_patch(plt.Rectangle((4, 4.5), 2, 1.2, color='white', alpha=0.9, edgecolor='darkgray', linewidth=0.5))
ax.text(5, 5.3, "2. REASONING", ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, 4.9, "(Fuse Data, Apply Logic)", ha='center', va='center', fontsize=8)
ax.annotate("", xy=(5, 4.5), xytext=(5, 3.8), arrowprops=dict(arrowstyle="->", color="darkgray", lw=1))

# Step 3: Action & Formatting
ax.add_patch(plt.Rectangle((4, 2.5), 2, 0.8, color='white', alpha=0.9, edgecolor='darkgray', linewidth=0.5))
ax.text(5, 2.9, "3. ACTION PLAN", ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, 2.7, "(Generate Output)", ha='center', va='center', fontsize=8)

# --- Output (Right) - The Agent Plan
ax.add_patch(plt.Rectangle((7, 4.5), 2.5, 2, color='#FFFF99', alpha=0.8, edgecolor='black', linewidth=1, zorder=2))
ax.text(8.25, 5.9, "Agent Plan Output", ha='center', va='center', fontsize=14, fontweight='bold', color='black')
ax.text(8.25, 5.5, "(Structured Steps for Tools)", ha='center', va='center', fontsize=10, color='black')
ax.text(8.25, 5.2, "e.g., Search Query, Click Sequence", ha='center', va='center', fontsize=8, color='darkgray')


# Arrow from Qwen3-VL to Output
ax.annotate("", xy=(6.3, 5.5), xytext=(7, 5.5), arrowprops=dict(arrowstyle="->", color="gray", lw=2))

ax.set_title("Agent Plan Demonstration: Qwen3-VL Thinking Model", fontsize=16, pad=20)

plt.savefig('agent_plan_demonstration.png', bbox_inches='tight', dpi=300)
print("agent_plan_demonstration.png")