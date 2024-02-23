import os
import sys
import threading

from importlib.util import find_spec


def find_python():
    if hasattr(sys, "frozen"):
        if sys.platform == "darwin":
            app_python = os.path.join(os.path.dirname(os.environ["ARGVZERO"]), "python")
            return app_python
    return sys.executable


def find_worker_env(idx):
    """Construct a command-line environment for the worker

    idx - index of the worker, e.g., 0 for the first, 1 for the second...
    """
    newenv = os.environ.copy()
    cp_install = find_spec("cellprofiler")
    if cp_install:
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(cp_install.origin), "..")
        )
    else:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    added_paths = []
    if "PYTHONPATH" in newenv:
        old_path = newenv["PYTHONPATH"]
        if not any([root_dir == path for path in old_path.split(os.pathsep)]):
            added_paths.append(root_dir)
    else:
        added_paths.append(root_dir)

    if hasattr(sys, "frozen"):
        if sys.platform == "darwin":
            # http://mail.python.org/pipermail/pythonmac-sig/2005-April/013852.html
            added_paths += [p for p in sys.path if isinstance(p, str)]
    if "PYTHONPATH" in newenv:
        added_paths.insert(0, newenv["PYTHONPATH"])
    newenv["PYTHONPATH"] = os.pathsep.join([x for x in added_paths])
    if "CP_JDWP_PORT" in newenv:
        del newenv["CP_JDWP_PORT"]
    if "AW_JDWP_PORT" in newenv:
        port = str(int(newenv["AW_JDWP_PORT"]) + idx)
        newenv["CP_JDWP_PORT"] = port
        del newenv["AW_JDWP_PORT"]
    for key in newenv:
        if isinstance(newenv[key], str):
            newenv[key] = newenv[key]
    return newenv


def find_analysis_worker_source():
    # import here to break circular dependency.
    import cellprofiler_core.worker  # used to get the path to the code

    return os.path.join(
        os.path.dirname(cellprofiler_core.worker.__file__), "__init__.py"
    )


def start_daemon_thread(target=None, args=(), kwargs=None, name=None):
    thread = threading.Thread(target=target, daemon=True, args=args, kwargs=kwargs, name=name)
    thread.start()
    return thread


def close_all_on_exec():
    """Mark every file handle above 2 with CLOEXEC

    We don't want child processes inheret anything
    except for STDIN / STDOUT / STDERR. This should
    make it so in a horribly brute-force way.
    """
    import fcntl

    try:
        maxfd = os.sysconf("SC_OPEN_MAX")
    except:
        maxfd = 256
    for fd in range(3, maxfd):
        try:
            fcntl.fcntl(fd, fcntl.FD_CLOEXEC)
        except:
            pass
