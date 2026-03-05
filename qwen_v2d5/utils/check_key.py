#!/usr/bin/env python3
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
