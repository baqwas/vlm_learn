import csv
import os
from collections import Counter


def analyze_vlm_tasks(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return

    tasks = []

    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            # Using DictReader to handle columns by name
            reader = csv.DictReader(f)
            for row in reader:
                tasks.append(row)

        # Count statuses (assumes a column named 'Status')
        status_counts = Counter(task.get("Status", "Unknown") for task in tasks)

        # Calculate percentages
        total = len(tasks)
        completed = status_counts.get("Completed", 0)
        pending = total - completed

        print("\n" + "=" * 40)
        print(f"📊 VLM PROJECT PROGRESS REPORT")
        print("=" * 40)
        print(f"✅ Completed Tasks: {completed}")
        print(f"⏳ Pending Tasks:   {pending}")
        print(f"📈 Total Tasks:     {total}")

        if total > 0:
            progress = (completed / total) * 100
            print(f"\nOverall Progress: {progress:.1f}%")

            # Simple progress bar
            bar_length = 20
            filled = int(bar_length * completed // total)
            bar = "█" * filled + "-" * (bar_length - filled)
            print(f"[{bar}]")
        print("=" * 40 + "\n")

    except KeyError:
        print("❌ Error: CSV must have a 'Status' column.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Pointing to your specific project timeline
    PATH = "projects/project_timeline.csv"
    analyze_vlm_tasks(PATH)
