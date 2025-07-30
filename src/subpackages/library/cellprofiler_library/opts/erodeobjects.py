# coding=utf-8

"""
Options and enums for ErodeObjects module
"""

from enum import Enum


class PreserveMidpoints(str, Enum):
    """Options for preserving object midpoints during erosion"""
    PREVENT_REMOVAL = "Yes"
    ALLOW_REMOVAL = "No"


class RelabelObjects(str, Enum):
    """Options for relabeling objects after erosion"""
    RELABEL = "Yes"
    KEEP_ORIGINAL = "No"
