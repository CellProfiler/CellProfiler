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
                # relying on Appose's bundled `pixi run` to pick one on its
                # own. If CellProfiler itself is running inside an
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


def run_python_task(environment, script, inputs=None):
    """
    Run `script` to completion in a new Python service backed by
    `environment`, passing `inputs` and returning the task's outputs.

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
    try:
        with environment.python() as python_env:
            task = python_env.task(script, inputs=converted_inputs)
            task.wait_for()
            return {
                key: _from_ndarray(value) if isinstance(value, appose.NDArray) else value
                for key, value in task.outputs.items()
            }
    finally:
        for ndarray in ndarray_inputs:
            ndarray.shm.dispose()
