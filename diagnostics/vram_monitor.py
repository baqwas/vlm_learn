#!/usr/bin/env python3
"""
================================================================================
PROJECT: BeUlta VRAM Sanity Monitor 📊
AUTHOR:  Matha Goram 🛠️
VERSION: 1.1.0
DESCRIPTION: Real-time GPU telemetry to prevent OOM crashes during VLM audits.
LOCATION: /diagnostics/
================================================================================
NOTES:
Fresh Start:
    Clear caches:
        sync && sudo sysctl -w vm.drop_caches=3
    Terminate background tensors:
        fuser -v /dev/nvidia*
    # Kill the PID if necessary
Run:
    Install the dependency:
        pip installl nvidia-ml-py3
    Run in a separate terminal:
        python3 diagnostics/vram_monitor.py
    Start the "auditor": e.g.
        PYTHONPATH=. python3 clip_exercise/clip_work/video_auditor.py
Observations:
Run this in a small terminal window tucked in the corner.
When you prompt the model:
    Watch the jump:
        If it jumps from ~3.8GB (idle model) to 5.6GB+
        immediately after you press Enter, it means
        your dynamic_fps is still pushing too many frames
        for the 3B model's attention mechanism to handle comfortably.
    The Result:
        If it crashes, check vram_audit.log. If the peak is
        near 5900 MB, you know exactly where the ceiling is.
================================================================================
"""

import time
import os
import pynvml
from datetime import datetime


def monitor_vram(interval=0.5):
    """
    Polls NVIDIA driver for metrics and logs the peak usage to 'vram_audit.log'.
    """
    log_file = "vram_audit.log"
    peak_used = 0
    peak_time = ""

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_name = pynvml.nvmlDeviceGetName(handle)

        print(f"🚀 Monitoring GPU: {device_name}")
        print(f"📝 Logging session peaks to: {log_file}")
        print("Press Ctrl+C to stop and save log.\n")
        print(f"{'Timestamp':<20} | {'VRAM Usage':<15} | {'Free':<10} | {'Temp':<5}")
        print("-" * 75)

        while True:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            used_mb = info.used / 1024**2
            total_mb = info.total / 1024**2
            free_mb = info.free / 1024**2
            pct_used = (used_mb / total_mb) * 100

            # Track the peak
            if used_mb > peak_used:
                peak_used = used_mb
                peak_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Alert threshold for BeUlta (90% of 6GB is ~5.4GB)
            alert = "⚠️  LOW HEADROOM!" if pct_used > 90 else ""

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(
                f"{timestamp:<20} | {used_mb:>6.0f}/{total_mb:>6.0f} MB | {free_mb:>6.0f} MB | {temp:>3}°C  {alert}",
                end="\r",
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n🏁 Monitoring stopped.")
        if peak_used > 0:
            log_entry = (
                f"\n--- SESSION SUMMARY ({datetime.now().strftime('%Y-%m-%d')}) ---\n"
                f"Device: {device_name}\n"
                f"Peak VRAM Usage: {peak_used:.2f} MB\n"
                f"Recorded At: {peak_time}\n"
                f"{'Status: CRITICAL' if peak_used > 5500 else 'Status: STABLE'}\n"
                + "-" * 40
                + "\n"
            )
            with open(log_file, "a") as f:
                f.write(log_entry)
            print(f"📑 Peak usage of {peak_used:.2f} MB written to {log_file}")

    except Exception as e:
        print(f"\n🤦‍♂️ Error accessing GPU metrics: {e}")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    # Ensure you have run: pip install nvidia-ml-py3
    monitor_vram()
