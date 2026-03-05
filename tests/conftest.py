#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: Pytest configuration to skip hardware/VLM checks in CI environments.
================================================================================
"""

import os
import pytest


def pytest_configure(config):
    """Register custom markers for VLM and Hardware checks."""
    config.addinivalue_line(
        "markers", "vlm_hardware: mark test as requiring local VLM hardware/weights"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests marked with 'vlm_hardware' if running
    inside a GitHub Actions environment.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        skip_vlm = pytest.mark.skip(
            reason="Skipping hardware check: GitHub Runner detected 🛡️"
        )
        for item in items:
            if "vlm_hardware" in item.keywords:
                item.add_marker(skip_vlm)
