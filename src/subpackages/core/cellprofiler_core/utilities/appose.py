"""Helpers for plugins that offload their `run()` logic to a separate
process/environment via Appose (https://docs.apposed.org).

A plugin using this module ships:
- a lightweight frontend `Module` file, importable in the default
  CellProfiler environment (only standard cellprofiler dependencies at module
  level)
- an environment spec file (e.g. `pixi.toml`) declaring the plugin's own,
  possibly heavy, dependencies;
- a worker script (plain Python source) that uses those dependencies and
  is only ever executed inside the Appose-managed subprocess.
"""

import logging
import os
import pathlib
import threading

import appose
import numpy

LOGGER = logging.getLogger(__name__)

# macOS `pythonw` (conda's python.app launcher) exports PYTHONEXECUTABLE,
# which every Appose subprocess inherits. Any venv-based interpreter that sees
# it (e.g. the isolated PEP 517 build venvs uv creates while pixi builds an
# sdist) ignores its own pyvenv.cfg, loses its site-packages, and fails with
# `No module named 'setuptools'`. It only matters at interpreter startup, so
# dropping it from this, already-running, process is safe.
os.environ.pop("PYTHONEXECUTABLE", None)

_environments = {}
_environments_lock = threading.Lock()

_services = {}
_services_lock = threading.Lock()

_CLOSE_TIMEOUT_SECONDS = 5.0


def _env_label(env_spec):
    """A short name for `env_spec` to prefix its build log lines with.

    Per the plugin layout convention, a spec lives at
    `<plugin_dir>/pixi.toml`, so the parent directory names the plugin.
    """
    return pathlib.Path(env_spec).parent.name or pathlib.Path(env_spec).name


def _build(env_spec, progress=None):
    """
    Build `env_spec` with Appose's build reporting wired up to console logging,
    and to `progress` if given.

    - progress: `(title, current, maximum)` phase events, e.g.
      ("Installing conda packages", 12, 27). A handful per build, so these
      go to INFO
    - output/error: raw lines from the `pixi` subprocess. In practice pixi
      writes everything to stderr and nothing to stdout. Note that
      registering *any* progress subscriber makes Appose add `-vv` to the
      install command (that's how its `PixiInstallMonitor` recognizes
      phase transitions), which inflates stderr to hundreds of
      lines of DEBUG-level pixi internals per build, so these go to
      DEBUG
    """
    label = _env_label(env_spec)

    def on_progress(title, current, maximum):
        if maximum > 0:
            LOGGER.info("%s: %s (%d/%d)", label, title, current, maximum)
        else:
            LOGGER.info("%s: %s", label, title)
        if progress is not None:
            try:
                progress(title, current, maximum)
            except Exception:
                # Never let a consumer's callback break the build: this
                # runs on Appose's own stderr-reader/polling threads.
                LOGGER.warning(
                    "Progress callback failed for %s", label, exc_info=True
                )

    builder = appose.file(env_spec)
    builder.subscribe_progress(on_progress)
    builder.subscribe_output(lambda line: LOGGER.info("%s: %s", label, line.rstrip()))
    builder.subscribe_error(lambda line: LOGGER.debug("%s: %s", label, line.rstrip()))
    LOGGER.info("Building Appose environment for %s from %s", label, env_spec)
    environment = builder.build()
    LOGGER.info("Appose environment for %s is ready", label)
    return environment


def get_environment(env_spec, progress=None):
    """
    Build (or reuse a cached) Appose environment from an environment spec
    file, e.g. a `pixi.toml`, `pyproject.toml`, or `environment.yml`.

    Appose itself caches the built environment on disk, keyed off of the
    spec's content, and only re-resolves dependencies when the spec
    changes. This fn additionally caches the in-memory `Environment` object
    per spec path within the current process, so repeated calls (e.g. once
    per image set) skip even that up-to-date check.

    Build progress is logged (see `_build()`) via `progress` callback.
    It's called on Appose's own background threads, not the caller's.
    It's not called at all when a cached environment is returned, since nothing is built.
    """
    env_spec = str(pathlib.Path(env_spec).resolve())
    with _environments_lock:
        environment = _environments.get(env_spec)
        if environment is None:
            environment = _build(env_spec, progress)
            try:
                environment = environment.activate("default")
            except NotImplementedError:
                pass
            _environments[env_spec] = environment
        return environment


def environment_exists(env_spec):
    """
    Reports whether an env has been created on disk for `env_spec`.
    Cheap and side-effect-free FS query. No build/sync is triggered.

    True for every Appose `EnvStatus` except `MISSING`: an environment
    directory that exists but is of a foreign type (`INCOMPATIBLE`), was
    not created by Appose (`EXTERNAL`, which also covers one left behind by
    a build that failed partway), or was created by Appose and is either
    out of date (`STALE`) or current (`CURRENT`) all count as existing.
    """
    env_spec = str(pathlib.Path(env_spec).resolve())
    return appose.file(env_spec).status() is not appose.EnvStatus.MISSING


def rebuild_environment(env_spec, progress=None):
    """
    Delete the environment previously built from `env_spec` and build it
    again from scratch, returning the fresh `Environment`.

    Prefer `get_environment()` for ordinary use: its `build()` already
    syncs an existing environment to the current spec incrementally. This
    is for the case where that isn't enough, e.g. a previous build that failed
    partway, which leaves artifacts tricking build() into thinking it's fully built.
    """
    env_spec = str(pathlib.Path(env_spec).resolve())
    # close any persistent service cached for this spec first, since its
    # worker process lives inside the directory about to be deleted
    close_service(env_spec)
    # drop the in-process `Environment` cache entry so the rebuilt
    # environment is the one subsequent `get_environment()` calls see
    with _environments_lock:
        _environments.pop(env_spec, None)
    LOGGER.info("Deleting Appose environment for %s", _env_label(env_spec))
    appose.file(env_spec).delete()
    return get_environment(env_spec, progress)


def get_service(env_spec):
    """Start (or reuse a cached) persistent Appose Python service for the
    environment built from `env_spec`.

    Unlike `run_python_task`'s default fresh-service-per-call behavior, a
    service returned here stays alive across many calls, so a worker script
    using `task.export(...)` (see `run_python_task`'s docstring) can keep an
    expensive resource, e.g. keeping a DL model "warm" to avoid model init.

    Callers may pass this same `env_spec` to `close_service()` once the
    service is no longer needed (e.g. a module's `post_run`), to shut it
    down promptly.
    """
    env_spec = str(pathlib.Path(env_spec).resolve())
    with _services_lock:
        service = _services.get(env_spec)
        if service is None or not service.is_alive():
            service = get_environment(env_spec).python()
            service.exit_timeout = _CLOSE_TIMEOUT_SECONDS
            service.start()
            _services[env_spec] = service
        return service


def close_service(env_spec, timeout=_CLOSE_TIMEOUT_SECONDS):
    """Close a persistent service previously obtained via `get_service()` for
    the same `env_spec`, if one is cached.

    This is not strictly a requirement as Appose itself
    tracks every started `Service` and shuts down any still running at
    process exit (closing it, then killing its whole worker process tree if
    it hasn't exited within `Service.exit_timeout` seconds.
    """
    env_spec = str(pathlib.Path(env_spec).resolve())
    with _services_lock:
        service = _services.pop(env_spec, None)
    if service is None or not service.is_alive():
        return
    service.close(timeout=timeout)


def _to_ndarray(value):
    """numpy array -> Appose shared-memory NDArray."""
    ndarray = appose.NDArray(dtype=value.dtype.name, shape=list(value.shape))
    ndarray.ndarray()[:] = value
    return ndarray


def _from_ndarray(value):
    """Appose NDArray data -> plain numpy array
    then, release shared memory
    """
    array = numpy.array(value.ndarray())
    value.shm.unlink_on_dispose(True)
    value.shm.dispose()
    return array


def run_python_task(script, inputs=None, environment=None, service=None):
    """
    Run `script` to completion in a Python service, passing `inputs` and
    returning the task's outputs. Only one of `environment` or `service`
    must be given.

    Pass `environment` (as returned by `get_environment()`) to get
    a fresh `environment.python()` service started for this call alone and
    shut down before returning.
    The cost is paying for a new worker process (and, for model-based plugins, a fresh
    model load) every time.

    Pass `service` (as returned by `get_service()`) to reuse it as-is (not closed)
    across calls, so the worker process, its imports, and anything it caches all
    survive between them.

    Raises `appose.service.TaskException` if the worker-side script fails,
    is canceled, or the worker process crashes.
    """
    if (environment is None) == (service is None):
        raise ValueError("Pass exactly one of `environment` or `service`")

    ndarray_inputs = []

    def convert_input(value):
        if not isinstance(value, numpy.ndarray):
            return value
        ndarray = _to_ndarray(value)
        ndarray_inputs.append(ndarray)
        return ndarray

    converted_inputs = {
        key: convert_input(value) for key, value in (inputs or {}).items()
    }

    def run_task(python_env):
        task = python_env.task(script, inputs=converted_inputs)
        task.wait_for()
        return {
            key: _from_ndarray(value) if isinstance(value, appose.NDArray) else value
            for key, value in task.outputs.items()
        }

    try:
        if service is not None:
            return run_task(service)
        with environment.python() as python_env:
            return run_task(python_env)
    finally:
        for ndarray in ndarray_inputs:
            ndarray.shm.dispose()
