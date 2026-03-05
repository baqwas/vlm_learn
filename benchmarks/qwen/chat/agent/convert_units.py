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
# -*- coding: utf-8 -*-
"""
A utility to convert numerical values between different systems of units.
This function can handle all standard, metric, and engineering units.
@see https://pint.readthedocs.io/en/stable/
@license MIT
"""

from pint import UnitRegistry

ureg = UnitRegistry()


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Converts a numerical value from one system of units to another.
    This function can handle all standard, metric, and engineering units.

    Args:
        value: The numerical value to convert (e.g., 25.0).
        from_unit: The unit to convert from (e.g., 'celsius', 'meters_per_second').
        to_unit: The unit to convert to (e.g., 'fahrenheit', 'miles_per_hour').

    Returns:
        A string containing the converted value and its new unit.
    """
    try:
        quantity = value * ureg(from_unit)
        converted_quantity = quantity.to(to_unit)
        return str(converted_quantity)
    except Exception as e:
        return f"convert_units: error during conversion: {e}"
