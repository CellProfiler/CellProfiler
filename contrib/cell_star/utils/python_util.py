# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import os


def package_path(filename, quoted=1):
    """
    Return relative (from cwd) path to file in package folder.
    In quotes.
    @param filename:
    @param quoted:
    """
    name = os.path.join(os.path.dirname(os.path.dirname(__file__)), filename).replace("\\\\", "\\")
    if quoted:
        return "\"" + name + "\""
    return name