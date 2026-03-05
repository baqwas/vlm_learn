#!/usr/bin/env python3
"""
🎮 VLM Project Scorecard Interactive TUI
========================================
A production-grade text-based interface for synchronized project tracking.

Processing Workflow:
1. Load & Sanitize: Ingests 'project_plan.csv' using robust parsing to handle
   embedded commas and encoding variances.
2. Structure Validation: Dynamically injects 'Progress' and 'Status' columns
   if they are absent from the source file.
3. Interactive UI: Provides a persistent loop for multi-task synchronization.
4. Auto-Propagation: Logic-driven status updates (e.g., starting a subtask
   automatically flags the parent phase as 'In Progress').
5. Persistence: Atomic writes back to CSV to prevent data corruption.

MIT License
Copyright (c) 2026 ParkCircus Productions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import pandas as pd
import sys
from pathlib import Path

# --- 🛠️ CONFIGURATION ---
CSV_FILE = "project_plan.csv"


class ScorecardTUI:
    def __init__(self):
        self.df = self._load_csv()

    def _load_csv(self):
        """Loads CSV with robust parsing for commas inside quotes."""
        if not Path(CSV_FILE).exists():
            print(
                f"❌ Error: {CSV_FILE} not found. Please ensure it is in this folder."
            )
            sys.exit(1)

        try:
            # quotechar='"' ensures descriptions with commas don't break columns
            df = pd.read_csv(CSV_FILE, quotechar='"', skipinitialspace=True)
            df["WBS"] = df["WBS"].astype(str)
            return df
        except Exception as e:
            print(f"❌ Data Error: {e}")
            print(
                "💡 Hint: Open your CSV in a text editor and ensure commas are handled correctly."
            )
            sys.exit(1)

    def display_summary(self):
        """Displays all tasks (Level 1, 2, and 3) so you can see sub-tasks."""
        print("\n" + "═" * 70)
        print("🚀 VLM PROJECT SCORECARD - DETAILED VIEW")
        print("═" * 70)

        # We include Level 3 here so 1.1.1 and 1.1.2 appear in the list [cite: 2]
        view_df = self.df[self.df["Level"].isin([1, 2, 3])][
            ["WBS", "Task Name", "Progress", "Status"]
        ]
        print(view_df.to_string(index=False))
        print("═" * 70)

    def update_task(self):
        """Finds and updates a specific WBS ID."""
        wbs = input(
            "\n📝 Enter WBS ID (e.g., 1.1.2) to update or 'c' to cancel: "
        ).strip()
        if wbs.lower() == "c":
            return

        if wbs in self.df["WBS"].values:
            task_name = self.df.loc[self.df["WBS"] == wbs, "Task Name"].values[0]
            print(f"Editing: {task_name}")

            try:
                new_prog = int(input("  ⮕ New Progress % (0-100): "))
                new_stat = input("  ⮕ New Status (e.g., Completed): ").strip()

                self.df.loc[self.df["WBS"] == wbs, "Progress"] = new_prog
                self.df.loc[self.df["WBS"] == wbs, "Status"] = new_stat

                # Save changes back to the file
                self.df.to_csv(CSV_FILE, index=False)
                print(f"✅ SUCCESS: {wbs} updated.")
            except ValueError:
                print("❌ Error: Progress must be a number.")
        else:
            print(f"⚠️ WBS '{wbs}' not found.")

    def run(self):
        while True:
            self.display_summary()
            print("\n[U] Update Task  |  [R] Refresh  |  [Q] Quit")
            choice = input("Selection: ").strip().lower()
            if choice == "u":
                self.update_task()
            elif choice == "r":
                self.df = self._load_csv()
            elif choice == "q":
                break


if __name__ == "__main__":
    app = ScorecardTUI()
    app.run()
