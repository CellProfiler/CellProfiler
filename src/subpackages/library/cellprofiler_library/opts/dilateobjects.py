# coding=utf-8

"""
Options and enums for DilateObjects module
"""

from enum import Enum


class HandleOverlaps(str, Enum):
    """Options for handling overlaps during dilation"""
    PRESERVE = "Preserve overlap"
    OUTLINE = "Outline overlap"

