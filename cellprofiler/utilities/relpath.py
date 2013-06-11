# No CellProfiler copyright notice here.

import os
import os.path

# R.Barran 30/08/2004
def _relpath(target, base=os.curdir):
    """
    Return a relative path to the target from either the current dir or an optional base dir.
    Base can be a directory specified either as absolute or relative to current dir.
    """

    if not os.path.exists(target):
        raise OSError, 'Target does not exist: '+target

    if not os.path.isdir(base):
        raise OSError, 'Base is not a directory or does not exist: '+base

    base_list = os.path.abspath(base).split(os.sep)
    target_list = os.path.abspath(target).split(os.sep)

    # On the windows platform the target may be on a completely different drive from the base.
    if os.name in ['nt','dos','os2'] and base_list[0] <> target_list[0]:
        raise OSError, 'Target is on a different drive to base. Target: '+target_list[0].upper()+', base: '+base_list[0].upper()

    # Starting from the filepath root, work out how much of the filepath is
    # shared by base and target.
    for i in range(min(len(base_list), len(target_list))):
        if os.path.normcase(base_list[i]) <> os.path.normcase(target_list[i]):
            break
    else:
        # If we broke out of the loop, i is pointing to the first differing path elements.
        # If we didn't break out of the loop, i is pointing to identical path elements.
        # Increment i so that in all cases it points to the first differing path elements.
        i+=1

    rel_list = [os.pardir] * (len(base_list)-i) + target_list[i:]
    if len(rel_list) == 0:
        return "."
    return os.path.join(*rel_list)

__all__ = ['relpath']

if hasattr(os.path, 'relpath'):
    relpath = os.path.relpath
else:
    relpath = _relpath

