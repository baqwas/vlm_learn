#!/usr/bin/env python3
"""
bulk_downloader.py
"""
import os
import requests
from pathlib import Path

# Handle TOML library differences based on Python version
try:
    import tomllib  # Built-in in Python 3.11+
except ImportError:
    import tomli as tomllib  # pip install tomli for older versions


def download_bulk_images():
    # 1. Load Configuration
    try:
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        print("❌ Error: config.toml not found.")
        return

    # Extracting variables from config
    access_key = config["unsplash"]["access_key"]
    base_dir = Path(config["paths"]["save_base_dir"])
    categories = config["categories"]
    img_count = config["settings"]["images_per_category"]
    w = config["settings"]["target_width"]
    h = config["settings"]["target_height"]

    if access_key == "YOUR_UNSPLASH_ACCESS_KEY_HERE":
        print("❌ Error: Please update your Access Key in config.toml")
        return

    headers = {"Authorization": f"Client-ID {access_key}"}

    # 2. Process Categories
    for folder_name, query in categories.items():
        target_dir = base_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 Sourcing {img_count} images for: {folder_name.upper()}...")

        url = "https://api.unsplash.com/search/photos"
        params = {"query": query, "per_page": img_count, "orientation": "squarish"}

        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            results = response.json().get("results", [])
            for i, item in enumerate(results):
                # Dynamically set resolution via URL parameters
                img_url = f"{item['urls']['raw']}&w={w}&h={h}&fit=crop"

                try:
                    img_data = requests.get(img_url).content
                    filename = f"{folder_name}_{i + 1:02d}.jpg"

                    with open(target_dir / filename, "wb") as handler:
                        handler.write(img_data)
                    print(f"  ✅ Saved: {filename}")
                except Exception as e:
                    print(f"  ⚠️ Failed to download image {i + 1}: {e}")
        else:
            print(f"  ❌ API Error {response.status_code}: {response.text}")

    print(f"\n🎉 Bulk download complete. Images are in: {base_dir}")


if __name__ == "__main__":
    # pip install requests tomli
    download_bulk_images()
