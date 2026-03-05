import pandas as pd
import matplotlib.pyplot as plt


def generate_vlm_dashboard(csv_file):
    # 1. Load the status data
    df = pd.read_csv(csv_file)

    # 2. Filter for high-level Phases (Level 1)
    phases = df[df['Level'] == 1].copy()
    phases = phases.sort_values('WBS', ascending=False)  # Reverse for display

    # 3. Setup Plotting
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color mapping for Status
    color_map = {
        'Completed': '#2ecc71',  # Green
        'In Progress': '#f1c40f',  # Yellow
        'Not Started': '#95a5a6'  # Grey
    }
    colors = [color_map.get(s, '#bdc3c7') for s in phases['Status']]

    # Create horizontal bars
    bars = ax.barh(phases['Task Name'], phases['Progress'], color=colors, edgecolor='white')

    # 4. Add labels and metrics
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{int(width)}%',
                va='center', fontweight='bold', color='#2c3e50')

    # Calculate overall project completion
    overall_completion = phases['Progress'].mean()

    # Formatting
    ax.set_xlim(0, 110)
    ax.set_xlabel('Completion Percentage (%)', fontweight='bold')
    ax.set_title(f'VLM Benchmarking Project Status\nOverall Progress: {overall_completion:.1f}%',
                 fontsize=16, fontweight='bold', pad=20)

    # 5. Add Milestone Summary Box
    milestone_stats = df[df['Level'] == 2]['Status'].value_counts()
    summary = "Milestone Summary:\n" + "\n".join([f"• {s}: {c}" for s, c in milestone_stats.items()])

    plt.text(1.05, 0.5, summary, transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', boxstyle='round,pad=1'),
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig('project_dashboard.png', dpi=300)
    print("Dashboard generated: project_dashboard.png")


# Execute
generate_vlm_dashboard('vlm_project_status.csv')
