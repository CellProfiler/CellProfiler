# Context: porting CellProfiler plugins to run via Appose

Handoff doc for a fresh session continuing this work. Written 2026-10-05.

## Goal

Some "officially supported" plugins in `CellProfiler-plugins/CP5/active_plugins/`
(sibling repo at `/Users/Nodar/Developer/CellProfiler/CellProfiler-plugins`)
require heavy/exotic Python dependencies that aren't in the default
CellProfiler environment (torch, cellpose, etc.). Today the user has to
manually install those deps into CellProfiler's own environment.

Plan: move the `run()` logic of such plugins into a separate process,
running in its own, plugin-owned, pixi-managed environment, launched via
[Appose](https://docs.apposed.org/en/latest/index.html). CellProfiler's
main process only gathers inputs from the workspace, ships them to the
subprocess, and unpacks the results — conceptually the same "pure logic in
one layer, workspace plumbing in another" split as the
`cellprofiler_library` refactor pattern (`REFACTOR_PATTERN.md` in this
repo), except the "other layer" here is a different OS process/environment
instead of just a different Python module.

## Key architectural finding (load-bearing — don't redo this research)

**CellProfiler's plugin loader needs no changes.** Traced in
`cellprofiler_core/utilities/core/plugins.py`:

- `plugin_list()` does a flat, non-recursive
  `glob.glob(plugin_dir + "/[!_]*.py")` — files in subdirectories, or
  leading-underscore files, are invisible to it automatically.
- `load_plugin()` does `importlib.import_module(source)` after
  `sys.path.insert(0, plugin_dir)`. Any `ImportError` anywhere in the
  plugin file's top-level code is caught by a bare `except Exception`,
  logged as a warning, and the plugin is **silently dropped** — no
  manifest/metadata mechanism exists, and it's strictly all-or-nothing (no
  lazy-import tolerance).

Since `appose` is already a pixi dependency of CellProfiler's own default
environment (`pixi.toml`, `[feature.mod.pypi-dependencies]`:
`appose = { git = "https://github.com/apposed/appose-python.git" }`,
unpinned — floats to whatever git main resolves to at lock time), a
plugin's **frontend** file loads fine in the default environment as long as
it only imports `cellprofiler_core`, `appose`, `numpy`, `pathlib`/stdlib at
top level. The actual heavy dependencies only need to exist inside the
plugin's own Appose-managed subprocess, never in the host process. This is
architectural quarantine, not a flag CellProfiler has to check for — no
"detect this is an Appose plugin" logic is needed or wanted anywhere in
`cellprofiler_core`.

## Design decisions already made

(Asked and answered via AskUserQuestion earlier in this project — don't
re-ask unless the user wants to revisit.)

1. **Bridge helper location**: `cellprofiler_core/utilities/appose.py` (not
   in `CellProfiler-plugins`). Reusable by any plugin in either repo, since
   `appose` is already a core dependency.
2. **Environment build strategy**: build on demand from a `pixi.toml`
   shipped next to the plugin, via `appose.file(path).build()` — not the
   older pattern in `appose_demo.py` (see below) which requires the user to
   manually `pixi install` a sub-project first. Self-contained: drop in the
   plugin + its `pixi.toml` and go.

## Per-plugin file layout convention (settled on, validated via PixelShuffle POC)

```
active_plugins/<name>.py            # frontend Module — thin, light imports only
active_plugins/<name>/pixi.toml     # plugin's OWN env spec (its real deps + appose itself)
active_plugins/<name>/_worker.py    # pure processing function(s); never imported by CellProfiler
```

- `<name>.py`: only imports `cellprofiler.*`, `cellprofiler_core.*`, `pathlib`, and
  `cellprofiler_core.utilities.appose` (`get_environment`,
  `run_python_task`). In `run()`: pull `pixel_data` etc. out of the
  workspace, call `get_environment(spec_path)` (cheap — cached), call
  `run_python_task(environment, script, inputs={...})`, unpack the
  returned dict into a `cellprofiler_core.image.Image(...)` and add it to
  the image set.
- `<name>/pixi.toml`: must declare `appose` itself (the worker subprocess
  runs `appose.python_worker`, Appose's own wire-protocol implementation,
  so it must be installed there too), plus whatever real/heavy deps the
  plugin needs. Keep `python` permissive (e.g. `>=3.9`) — host and worker
  are fully separate interpreters/processes, their Python versions don't
  need to match.
- `<name>/_worker.py`: pure functions, zero `cellprofiler_core` imports —
  this file is read as **plain text** by the frontend `.py` file and
  concatenated with a small inline "glue" epilogue (currently just a
  string built in the frontend file — see Gotcha 2 below) that converts
  the raw input `appose.NDArray` to a numpy array, calls the worker
  function, wraps the result back into a fresh `NDArray`, and assigns it to
  `task.outputs["<key>"]`. This glue isn't yet factored into a reusable
  template in the bridge module — only one plugin (now reverted) has gone
  through this, so it wasn't necessary. Worth doing if/when porting a
  second plugin.

## The bridge: `cellprofiler_core/utilities/appose.py`

```python
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
```

### How a frontend plugin file calls it (pattern, from the now-reverted PixelShuffle POC)

```python
import pathlib
import cellprofiler_core.image
import cellprofiler_core.module
import cellprofiler_core.setting
from cellprofiler_core.utilities.appose import get_environment, run_python_task

_PLUGIN_DIR = pathlib.Path(__file__).parent / "pixelshuffle"
_ENV_SPEC = _PLUGIN_DIR / "pixi.toml"
_WORKER_SCRIPT = (_PLUGIN_DIR / "_worker.py").read_text() + (
    "import appose\n"
    "import numpy\n"
    "_result = pixel_shuffle(numpy.array(x_data.ndarray()))\n"
    "_result_ndarray = appose.NDArray(dtype=_result.dtype.name, shape=list(_result.shape))\n"
    "_result_ndarray.ndarray()[:] = _result\n"
    'task.outputs["pixel_data"] = _result_ndarray\n'
)

class PixelShuffle(cellprofiler_core.module.ImageProcessing):
    ...
    def run(self, workspace):
        x = workspace.image_set.get_image(self.x_name.value)
        environment = get_environment(_ENV_SPEC)
        outputs = run_python_task(environment, _WORKER_SCRIPT, inputs={"x_data": x.pixel_data})
        y_data = outputs["pixel_data"]
        workspace.image_set.add(self.y_name.value, cellprofiler_core.image.Image(
            dimensions=x.dimensions, image=y_data, parent_image=x
        ))
```

And the plugin's `pixi.toml` (conda-forge, NOT a git dependency on appose — see
Gotcha 1):

```toml
[workspace]
authors = ["..."]
channels = ["conda-forge"]
name = "pixelshuffle"
platforms = ["osx-64", "osx-arm64", "linux-64", "win-64"]
version = "0.1.0"

[dependencies]
python = ">=3.9"
numpy = "*"
appose = ">=0.12,<0.13"
```

## Gotchas discovered (all load-bearing, re-derive-costly — don't rediscover)

### 1. Host vs. worker Appose version skew

CellProfiler's own `appose` pin floats to git main (currently resolves to
something like `0.12.1.dev0`, with newer API: `NDArray.copy_of(arr)`,
`numpy.asarray(ndarray)` via `__array__`, `PixiBuilder`/`DynamicBuilder`,
etc.). A plugin's own `pixi.toml` pulling `appose` from **conda-forge**
will get the latest *released* version (0.12.0 at time of writing), which
**lacks** `copy_of`/`__array__` — only has the older, manual
`appose.NDArray(dtype=..., shape=...)` constructor + `.ndarray()` method.

**Rule: always use only the oldest-common-denominator NDArray API**
(manual construct + `.ndarray()`) in both the bridge (`appose.py`) and any
worker glue code — never `.copy_of()` or `numpy.asarray(an_ndarray)` —
since host and worker environments are built independently and will not
necessarily have matching Appose releases.

Also: the installed appose package in `.pixi/envs/dev/lib/python3.9/site-packages/appose/`
is the ground truth for "what API is actually available right now" — a
sibling loose checkout at `/Users/Nodar/Developer/CellProfiler/appose-python`
is **not** reliable as a reference (its HEAD, `79dd2cd1`, is confusingly an
older/different snapshot than what's actually installed via the floating
git pin). Don't read that checkout's source and assume it matches; always
check the installed package directly, or pull fresh.

Do **not** try to pin a plugin's `pixi.toml` to the same git source as
CellProfiler's own `appose` dependency to "fix" this — it was tried, and
pixi's git-dependency fetch is unreliable inside this sandboxed environment
(intermittent `io error: unexpected end of file` during `pixi install`,
root cause not fully isolated, plausibly a pixi-specific git transport
issue since plain `git clone` of the same URL works fine). The
conda-forge release + lowest-common-denominator API approach avoids this
entirely and is simpler anyway.

### 2. `appose` must be a dependency of the plugin's own environment too

The worker subprocess runs `appose.python_worker` (Appose's wire-protocol
implementation) — it needs `appose` installed in the *plugin's* pixi
environment, not just the host's. Easy to forget since the host already
has it.

### 3. `pixi` bug: ambient `PIXI_IN_SHELL`+`PIXI_ENVIRONMENT_NAME` leak into unrelated `pixi run --manifest-path`

**This is the one that actually broke things in real usage (reported by
the user running inside the GUI from a `pixi run -e dev`/`pixi shell`
session) and took the most effort to root-cause.**

Appose's `PixiBuilder` shells out to a bundled `pixi` binary as
`pixi run --manifest-path <plugin-pixi.toml> <python-exe> ...` to build
and launch the plugin's own, unrelated pixi project. If the *calling*
process (CellProfiler) is itself running inside an activated, named pixi
environment (this repo's `dev` environment, via `pixi run -e dev` /
`pixi shell`), pixi sets `PIXI_IN_SHELL=1` and `PIXI_ENVIRONMENT_NAME=dev`
in the process environment, and both leak into Appose's subprocess call.
`pixi` then — reproducibly, and apparently a genuine `pixi` bug —
**trusts the ambient `PIXI_ENVIRONMENT_NAME` for environment selection
whenever `PIXI_IN_SHELL=1` is also set, even when an explicit
`--manifest-path` names a completely different project** that has no
same-named environment (only the implicit `default`). Fails with:

```
Error:   × unknown environment 'dev'
```

Bisected carefully (see full writeup, still on disk, at
`/Users/Nodar/Developer/CellProfiler/appose-python/PIXI_ENV_NAME_LEAK_REPRO.md`
— that's a loose sibling checkout, not part of either git repo, so it
won't survive a disk wipe; copy it out if it matters long-term):

- Neither `PIXI_IN_SHELL=1` alone nor `PIXI_ENVIRONMENT_NAME=dev` alone
  triggers it — **both together** are required.
- Reproduces identically on pixi 0.58.0 (the version Appose's
  `PixiBuilder` bundles/downloads for itself,
  `appose/tool/pixi.py:PIXI_VERSION`) and pixi 0.81.0 (current release at
  time of writing). Pure `pixi` CLI behavior, nothing to do with Appose or
  CellProfiler code.
- **Fix that works**: passing `--environment default` explicitly on the
  `pixi run` command line sidesteps the bug entirely (tested: succeeds
  even with the ambient vars set).

**Applied fix** (already in the bridge code above, `get_environment()`):
after `appose.file(env_spec).build()`, call
`environment.activate("default")`, which is Appose's own supported API for
getting an `Environment` whose `launch_args()` include
`["--environment", "default"]` baked in — see
`appose/environment.py:Environment.activate()` and
`appose/builder/pixi.py:_build_pixi_environment`'s `activator` closure.
Wrapped in `try/except NotImplementedError` because not every Appose
builder/environment type supports named sub-environments (e.g. a plain
`appose.base(prebuilt_dir)`-wrapped environment, like `appose_demo.py`
uses, doesn't).

Verified fixed by running the full PixelShuffle round trip from inside the
actual CellProfiler `dev` pixi shell (the exact failure conditions the user
hit) — passed cleanly after the `.activate("default")` fix, no env-var
stripping workaround needed.

This is worth filing upstream against `pixi` itself — the repro doc has
everything needed for a bug report. The `.activate("default")` fix in our
bridge only neutralizes it for Appose-managed environments; it doesn't fix
`pixi` generally.

## Existing reference code in CellProfiler-plugins (pre-existing, not ours)

In `CellProfiler-plugins/CP5/active_plugins/`:

- **`cpij/`**: an older, pre-Appose, unrelated `multiprocessing.managers`-based
  ImageJ bridge. Not relevant to this work, just legacy context explaining
  why an `appose_demo.py` exists alongside it.

## Current on-disk state (as of this writing)

- `CellProfiler-plugins` repo: **clean**.
    * The PixelShuffle port (frontend
      rewrite + `pixelshuffle/pixi.toml` + `pixelshuffle/_worker.py`) was
      reverted by the user after confirming the approach worked — it was a
      disposable proof of concept, not meant to be kept as a PR.
    - **`appose_demo.py`** (also reverted):
      a working `Module` demoing Appose + napari running
      in a subprocess. Uses `appose.base(prebuilt_env_dir).build()` — points at
      an **already pixi-installed** env directory, requiring the user to
      manually run `pixi install` first (unlike our on-demand
      `appose.file(pixi_toml).build()` approach). Demonstrates
      `python.task(script, queue="main")` for a long-lived Qt/napari task, and
      the same manual `NDArray` construct + `.ndarray()[:] = ...` pattern we
      ended up standardizing on for portability.
    - **`apposednapari/`** (also reverted):
      the pixi sub-project (`pixi.toml` + `pyproject.toml`)
      `appose_demo.py` points at.
- `CellProfiler` repo: `src/subpackages/core/cellprofiler_core/utilities/appose.py`
  **still exists on disk**, untracked (`git status` shows `A`, not yet
  committed). This is the one real deliverable artifact — full contents
  above. This doc (`APPOSE_PLUGINS_CONTEXT.md`) is also untracked, sitting
  at the CellProfiler repo root.
- `/Users/Nodar/Developer/CellProfiler/appose-python/PIXI_ENV_NAME_LEAK_REPRO.md`
  exists in the loose sibling checkout (not a tracked repo artifact).

## Open questions / not yet decided

1. **Which plugin to port for real** (PixelShuffle was just the POC/vehicle
   to validate the pattern and shake out the bugs above — it has no real
   heavy dependency of its own, so it doesn't actually *need* Appose; a
   good "real" first candidate would be something that currently requires
   manual dependency installation, e.g. something using `torch`/`cellpose`/
   `omnipose`/`stardist` in `active_plugins/`).
2. **Service lifecycle / performance**: `run_python_task` currently opens a
   **fresh** `environment.python()` service per call (simplest, safest
   resource semantics — mirrors the `with` pattern in `appose_demo.py`).
   For a plugin processing many image sets in a real pipeline run, this
   means a fresh process launch per image, which may be too slow. Keeping
   one persistent `Service` alive for a module instance's lifetime would
   need a CellProfiler module lifecycle hook to close it at the end of a
   run (not investigated — look at whether `Module` has something like
   `post_run`/`on_deactivated`).
3. **Worker glue templating**: the "convert input NDArray → numpy → call
   worker function → wrap output back into NDArray → assign to
   `task.outputs[...]`" glue currently has to be hand-written per plugin as
   a string concatenated onto the `_worker.py` source. Worth factoring into
   a small reusable template/helper in `cellprofiler_core/utilities/appose.py`
   once a second plugin is ported and the shape of the duplication is
   clearer.
4. **Committing the bridge file**: `cellprofiler_core/utilities/appose.py`
   is untracked — decide whether/when to commit it (probably once there's
   at least one real plugin PR using it, so it lands with a concrete
   consumer rather than as speculative infra).
5. File the `pixi` bug upstream (ambient `PIXI_IN_SHELL`+`PIXI_ENVIRONMENT_NAME`
   overriding `--manifest-path` environment resolution) — not yet done.
