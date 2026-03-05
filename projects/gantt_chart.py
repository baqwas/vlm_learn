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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def generate_vlm_gantt(csv_file):
    df = pd.read_csv(csv_file)
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"] = pd.to_datetime(df["End"])
    df["Duration"] = (df["End"] - df["Start"]).dt.days

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Status-based colors
    color_map = {
        "Completed": "#2ecc71",
        "In Progress": "#f1c40f",
        "Not Started": "#bdc3c7",
    }

    # Plot each bar
    for i, row in df.iterrows():
        ax.barh(
            row["Task Name"],
            row["Duration"],
            left=row["Start"],
            color=color_map.get(row["Status"]),
            edgecolor="black",
        )

    # Add 'Today' line
    ax.axvline(datetime.now(), color="red", linestyle="--", label="Today")

    # Format Date Axis
    ax.invert_yaxis()  # Tasks top-to-bottom
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.title("VLM Benchmarking Project Schedule", fontsize=15, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vlm_gantt_chart.png")
