"""Helpers for plugins that offload their `run()` logic to a separate
process/environment via Appose (https://docs.apposed.org).

A plugin using this module ships:
- a lightweight frontend `Module` file, importable in the default
  CellProfiler environment (only `cellprofiler_core`, `appose`, `numpy`,
  and standard library at module level);
- an environment spec file (e.g. `pixi.toml`) declaring the plugin's own,
  possibly heavy, dependencies;
- a worker script (plain Python source) that uses those dependencies and
  is only ever executed inside the Appose-managed subprocess, never
  imported by the main CellProfiler process.

No changes to CellProfiler's plugin discovery/import mechanism are needed:
since the frontend file's own top-level imports are always satisfiable in
the default environment, it loads like any other plugin.
"""

import pathlib
import threading

import appose
import numpy

_environments = {}
_environments_lock = threading.Lock()

_services = {}
_services_lock = threading.Lock()

_CLOSE_TIMEOUT_SECONDS = 5.0


def get_environment(env_spec):
    """
    Build (or reuse a cached) Appose environment from an environment spec
    file, e.g. a `pixi.toml`, `pyproject.toml`, or `environment.yml`.

    Appose itself caches the built environment on disk, keyed off of the
    spec's content, and only re-resolves dependencies when the spec
    changes. This additionally caches the in-memory `Environment` object
    per spec path within the current process, so repeated calls (e.g. once
    per image set) skip even that up-to-date check.
    """
    env_spec = str(pathlib.Path(env_spec).resolve())
    with _environments_lock:
        environment = _environments.get(env_spec)
        if environment is None:
            environment = appose.file(env_spec).build()
            try:
                # Explicitly target the "default" environment rather than
                # relying on Appose's bundled `pixi run` to pick one on its own.
                # Why?: https://github.com/prefix-dev/pixi/issues/7170
                # TL;DR: If CellProfiler itself is running inside an
                # activated, named pixi environment (e.g. this repo's own
                # `dev` environment), `PIXI_IN_SHELL`/`PIXI_ENVIRONMENT_NAME`
                # leak into that subprocess and pixi tries to find a
                # same-named environment in the *plugin's* unrelated
                # project, failing with "unknown environment '<name>'".
                # Passing --environment default explicitly sidesteps that.
                environment = environment.activate("default")
            except NotImplementedError:
                pass
            _environments[env_spec] = environment
        return environment


def get_service(env_spec):
    """
    Start (or reuse a cached) persistent Appose Python service for the
    environment built from `env_spec`.

    Unlike `run_python_task`'s default fresh-service-per-call behavior, a
    service returned here stays alive across many calls, so a worker script
    using `task.export(...)` (see `run_python_task`'s docstring) can keep an
    expensive resource - a loaded ML model, for instance - warm in the
    worker process between calls instead of re-creating it every time.

    Callers may pass this same `env_spec` to `close_service()` once the
    service is no longer needed (e.g. a module's `post_run`), to shut it
    down promptly. This is a courtesy, not a requirement: Appose itself
    tracks every started `Service` and shuts down any still running at
    process exit (closing it, then killing its whole worker process tree if
    it hasn't exited within `Service.exit_timeout` seconds - see
    `close_service()`)
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
    """
    Close a persistent service previously obtained via `get_service()` for
    the same `env_spec`, if one is cached. Safe to call even if no service
    was ever created for this spec, or if it's already been closed.

    Delegates to `Service.close(timeout=...)`: asks the worker to shut down
    gracefully (by closing its stdin), waits up to `timeout` seconds, then
    kills its whole process tree - not just the immediately launched
    process, which matters for a `PixiBuilder`-built environment, whose
    worker is a child of a `pixi run` launcher - if it hasn't exited by then.
    """
    env_spec = str(pathlib.Path(env_spec).resolve())
    with _services_lock:
        service = _services.pop(env_spec, None)
    if service is None or not service.is_alive():
        return
    service.close(timeout=timeout)


def _to_ndarray(value):
    """Copy a numpy array into a new Appose shared-memory NDArray."""
    ndarray = appose.NDArray(dtype=value.dtype.name, shape=list(value.shape))
    ndarray.ndarray()[:] = value
    return ndarray


def _from_ndarray(value):
    """Copy an Appose NDArray's data out into a plain numpy array, then
    release its shared memory (the host attached to, but did not create,
    this segment, so disposal alone would only close it, not free it)."""
    array = numpy.array(value.ndarray())
    value.shm.unlink_on_dispose(True)
    value.shm.dispose()
    return array


def run_python_task(script, inputs=None, environment=None, service=None):
    """
    Run `script` to completion in a Python service, passing `inputs` and
    returning the task's outputs. Exactly one of `environment` or `service`
    must be given.

    By default, pass `environment` (as returned by `get_environment()`): a
    fresh `environment.python()` service is started for this call alone and
    shut down before returning - the simplest, safest option, at the cost of
    paying for a new worker process (and, for model-based plugins, a fresh
    model load) every time. For a module invoked once per image set in a
    real pipeline run, this can be the dominant cost.

    To amortize that cost, pass `service` (as returned by `get_service()`)
    instead: it is reused as-is (not closed) across calls, so the worker
    process, its imports, and anything it caches all survive between them.
    To actually keep an expensive resource warm across calls, the worker
    script must opt in explicitly with `task.export(name=value)` - plain
    variables/imports in the script do NOT automatically persist to the
    next call, since each task resets its global name space to only
    `task` plus whatever was previously exported. A typical pattern in a
    worker script:

        _model_cache_key = (model_name, weights_path)
        if globals().get("_cache_key") != _model_cache_key:
            _model = load_model(model_name, weights_path)
            task.export(_cache_key=_model_cache_key, _model=_model)
        # ... use `_model` ...

    recomputing/re-exporting only when the relevant settings actually
    change, so settings edits (e.g. in the GUI's test/debug mode, which
    reuses the same module instance and the same service across repeated,
    manually-triggered runs) are picked up correctly instead of silently
    using a stale cached resource.

    Any `numpy.ndarray` values in `inputs` are copied into Appose shared
    memory so the worker process can read them without a serialization
    round-trip; any `appose.NDArray` values among the task's outputs are
    copied back into plain `numpy.ndarray` in the returned dict. All shared
    memory allocated for the call, on either side, is released before
    returning.

    Uses only the lowest-common-denominator Appose `NDArray` API
    (construct + `.ndarray()`), since the host and worker environments are
    built independently and may end up with different Appose releases.

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
