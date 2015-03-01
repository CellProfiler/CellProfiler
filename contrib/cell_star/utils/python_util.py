# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import os

PROFILE_SPEED = False
PROFILE_MEMORY = False

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

try:
    import line_profiler
    import memory_profiler
except:
    pass

def speed_profile(func):
    def profiled_func(*args, **kwargs):
        try:
            profiler = line_profiler.LineProfiler()
            profiler.add_function(func)
            profiler.enable_by_count()
            return func(*args, **kwargs)
        finally:
            profiler.print_stats()
    if PROFILE_SPEED:
        return profiled_func
    else:
        return func



def memory_profile(func):
    if not PROFILE_MEMORY:
        return func
    else:
        if func is not None:
            def wrapper(*args, **kwargs):
                if PROFILE_MEMORY:
                    prof = memory_profiler.LineProfiler()
                    val = prof(func)(*args, **kwargs)
                    memory_profiler.show_results(prof)
                else:
                    val = func(*args, **kwargs)
                return val
            return wrapper
        else:
            def inner_wrapper(f):
                return memory_profiler.profile(f)
            return inner_wrapper

