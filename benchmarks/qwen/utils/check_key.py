#!/usr/bin/env python3
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
"""
check_key.py
"""
import os
from dotenv import load_dotenv

# This loads the variables from .env into the environment
load_dotenv()

key = os.getenv("GEMINI_API_KEY")

if key:
    print(f"✅ Success! Key found: {key[:8]}...")
else:
    print("❌ Failure: Key not found in .env file.")
