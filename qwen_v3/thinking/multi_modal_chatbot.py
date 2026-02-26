import matplotlib.pyplot as plt
import numpy as np
import io

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off') # Hide axes

# User Bubble (Left)
ax.add_patch(plt.Rectangle((0.5, 7.5), 2, 1, color='#ADD8E6', alpha=0.8, edgecolor='black', linewidth=0.5, zorder=2))
ax.text(1.5, 8, "User Input", ha='center', va='center', fontsize=12, fontweight='bold', color='black')
ax.text(1.5, 7.7, "(Text or Image)", ha='center', va='center', fontsize=9, color='black')
ax.plot([2.5, 3.5], [8, 8], color='gray', linestyle='--', linewidth=1) # Arrow to VLM

# Qwen3-VL Core (Center)
ax.add_patch(plt.Rectangle((3.5, 3), 3, 4, color='#90EE90', alpha=0.8, edgecolor='black', linewidth=1, zorder=2))
ax.text(5, 5.5, "Qwen3-VL", ha='center', va='center', fontsize=16, fontweight='bold', color='darkgreen')
ax.text(5, 4.8, "Multimodal Chatbot", ha='center', va='center', fontsize=10, color='darkgreen')

# Internal Process Arrows
ax.annotate("", xy=(5, 7.2), xytext=(5, 6.7), arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))
ax.text(5, 7.4, "Visual Understanding", ha='center', va='center', fontsize=8, color='darkgreen')

ax.annotate("", xy=(5, 3.8), xytext=(5, 3.3), arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))
ax.text(5, 4, "Language Processing", ha='center', va='center', fontsize=8, color='darkgreen')

ax.text(5, 6, "Fusion & Reasoning", ha='center', va='center', fontsize=10, color='darkgreen', fontweight='bold')


# Output Bubble (Right)
ax.add_patch(plt.Rectangle((7.5, 4.5), 2, 1, color='#ADD8E6', alpha=0.8, edgecolor='black', linewidth=0.5, zorder=2))
ax.text(8.5, 5, "Chatbot Response", ha='center', va='center', fontsize=12, fontweight='bold', color='black')
ax.plot([6.5, 7.5], [5, 5], color='gray', linestyle='--', linewidth=1) # Arrow from VLM

# Loop Back Arrow (Real-time aspect)
ax.annotate("", xy=(8.5, 4.5), xytext=(8.5, 2.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1, linestyle=':'))
ax.annotate("", xy=(1.5, 2.5), xytext=(1.5, 4.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1, linestyle=':'))
ax.annotate("", xy=(1.5, 7.5), xytext=(1.5, 2.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1, linestyle=':'))

# Add text to denote "Real-time"
ax.text(5, 1.5, "Real-time, Iterative Conversation", ha='center', va='center', fontsize=10, color='gray')


ax.set_title("Real-Time Multimodal Chatbot powered by Qwen3-VL", fontsize=16, pad=20)

plt.savefig('realtime_multimodal_chatbot.png', bbox_inches='tight', dpi=300)
print("realtime_multimodal_chatbot.png")