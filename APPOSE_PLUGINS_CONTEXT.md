# Context: porting CellProfiler plugins to run via Appose

Handoff doc for a fresh session continuing this work. Last updated 2026-10-06.

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

## Design decisions made

1. **Bridge helper location**: `cellprofiler_core/utilities/appose.py`
   (committed). Reusable by any plugin in either repo, since `appose` is
   already a core dependency.
2. **Environment build strategy**: build on demand from a `pixi.toml`
   shipped next to the plugin, via `appose.file(path).build()`.
   Self-contained: drop in the plugin + its `pixi.toml` and go.
3. **Worker glue dispatch**: rather than baking per-call string
   substitution into the glue appended to a plugin's `_worker.py` (as the
   original PixelShuffle POC did), pass a plain string/scalar `model` (or
   similar) task input and have a static, module-level glue epilogue
   `if`/`elif` branch on it at runtime. The glue string itself is then
   computed once at import time instead of being rebuilt per `run()` call.
   See `cellcast.py` for the pattern.

## Per-plugin file layout convention

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
  returned dict into a `cellprofiler_core.image.Image(...)` or
  `cellprofiler_core.object.Objects()` and add it to the image/object set.
- `<name>/pixi.toml`: must declare `appose` itself (the worker subprocess
  runs `appose.python_worker`, Appose's own wire-protocol implementation,
  so it must be installed there too), plus whatever real/heavy deps the
  plugin needs. Keep `python` permissive (e.g. `>=3.9`) — host and worker
  are fully separate interpreters/processes, their Python versions don't
  need to match. If a dependency isn't on conda-forge, declare it under
  `[pypi-dependencies]` instead — this now works fine (see Gotchas, pixi
  version).
- `<name>/_worker.py`: pure functions, zero `cellprofiler_core` imports —
  this file is read as **plain text** by the frontend `.py` file and
  concatenated with a small inline "glue" epilogue (see design decision #3
  above) that converts the raw input `appose.NDArray` to a numpy array,
  dispatches to the right worker function, wraps the result back into a
  fresh `NDArray`, and assigns it to `task.outputs["<key>"]`.

## Reference implementation: `RunCellcast`

`CellProfiler-plugins/CP5/active_plugins/cellcast.py` (+ `cellcast/pixi.toml`,
`cellcast/_worker.py`) is the first real plugin ported with this pattern —
wraps the [cellcast](https://github.com/uw-loci/cellcast) project's
StarDist2D (fluo + H&E) and StarDist3D (fluo) models. Verified end-to-end
(real `pixi.toml` build, real Appose subprocess, all three model paths) —
treat it as the canonical example when porting the next plugin, alongside
the bridge module's own docstrings.

Notable cellcast-specific details, in case they come up again:

- `cellcast` ships PyPI-only wheels (cp38-abi3), not on conda-forge — hence
  the `[pypi-dependencies]` entry in its `pixi.toml`.
- `cellcast.models.StarDist2D`/`StarDist3D` methods (`init_fluo`, `init_he`,
  `predict_fluo`, `predict_he`) return/accept plain numpy arrays; labels
  come back as `uint64`. Weights auto-download to `~/.cache/cellcast/weights/`
  on first use and are reused across subsequent worker-process launches.
- `StarDist3D.predict_fluo` needs a reasonably-sized Z-stack (an 8-plane
  test input triggered a Rust-side panic/`InvalidParameterEmptyArray`; 32
  planes worked fine) — not a bug in our plugin, just a model input-size
  floor worth knowing about if a user reports a crash on tiny stacks.

## The bridge: `cellprofiler_core/utilities/appose.py`

Committed (not reproduced here — read the file directly; it's short and
its own docstrings cover the API). Key behaviors to remember:

- `get_environment(env_spec)` builds (or reuses a cached) Appose
  `Environment` from a spec file, and tries `environment.activate("default")`
  (swallowing `NotImplementedError` for builders that don't support named
  sub-environments) — see Gotcha 2 below for why.
- `run_python_task(environment, script, inputs=None)` copies any
  `numpy.ndarray` inputs into Appose shared memory, runs `script` in a
  **fresh** `environment.python()` service, copies any `appose.NDArray`
  outputs back into plain numpy arrays, and releases all shared memory
  before returning (see Open Questions for the "fresh service per call"
  performance tradeoff).
- Both only use the oldest-common-denominator Appose `NDArray` API (manual
  `appose.NDArray(dtype=..., shape=...)` construct + `.ndarray()`) — see
  Gotcha 1.

## Gotchas discovered (load-bearing, re-derive-costly — don't rediscover)

### 1. Host vs. worker Appose version skew

CellProfiler's own `appose` pin floats to git main, which may have a newer
API (`NDArray.copy_of(arr)`, `numpy.asarray(ndarray)` via `__array__`,
etc.) than whatever a plugin's own `pixi.toml` resolves (e.g. conda-forge's
latest *release*, which lacks those newer methods).

**Rule: always use only the oldest-common-denominator NDArray API**
(manual construct + `.ndarray()`) in both the bridge (`appose.py`) and any
worker glue code — never `.copy_of()` or `numpy.asarray(an_ndarray)` —
since host and worker environments are built independently and will not
necessarily have matching Appose releases.

The installed appose package under `.pixi/envs/dev/lib/python3.9/site-packages/appose/`
is the ground truth for "what API is actually available right now" in the
host — check it directly (or pull fresh) rather than trusting a loose
sibling checkout's HEAD, which can silently be a different snapshot.

### 2. `pixi` bug: ambient `PIXI_IN_SHELL`+`PIXI_ENVIRONMENT_NAME` leak into unrelated `pixi run --manifest-path`

Appose's `PixiBuilder` shells out to its bundled `pixi` binary as
`pixi run --manifest-path <plugin-pixi.toml> <python-exe> ...` to build and
launch the plugin's own, unrelated pixi project. If the *calling* process
(CellProfiler) is itself running inside an activated, named pixi
environment (e.g. this repo's `dev` environment, via `pixi run -e dev` /
`pixi shell`), pixi sets `PIXI_IN_SHELL=1` and `PIXI_ENVIRONMENT_NAME=dev`,
and both leak into Appose's subprocess call. `pixi` then — reproducibly,
confirmed as a genuine `pixi` bug — trusts the ambient
`PIXI_ENVIRONMENT_NAME` for environment selection whenever `PIXI_IN_SHELL=1`
is also set, even when an explicit `--manifest-path` names a completely
different project with no same-named environment. Fails with
`Error: × unknown environment 'dev'`.

**Filed upstream**: <https://github.com/prefix-dev/pixi/issues/7170>.

**Fix applied** in the bridge's `get_environment()`: call
`environment.activate("default")` after building, which bakes
`["--environment", "default"]` into the environment's `launch_args()`,
sidestepping the bug entirely. Wrapped in `try/except NotImplementedError`
since not every builder/environment type supports named sub-environments.
Verified fixed by running a full round trip from inside the actual
CellProfiler `dev` pixi shell (the exact failure conditions originally hit).

### 3. `appose` must be a dependency of the plugin's own environment too

The worker subprocess runs `appose.python_worker` (Appose's wire-protocol
implementation) — it needs `appose` installed in the *plugin's* pixi
environment, not just the host's. Easy to forget since the host already
has it.

### 4. Appose's bundled pixi version matters — pin to `main`, not an old release

Appose's `PixiBuilder` doesn't use whatever `pixi` is on `$PATH`; it
downloads and pins its own copy (`appose/tool/pixi.py:PIXI_VERSION`). The
pin was `v0.58.0` for a long time, which has a real, deterministic bug:
`pixi install` fails with `io error: unexpected end of file` on **any**
`[pypi-dependencies]` entry (reproduced even with a trivial package like
`six` — nothing cellcast-specific). This was fixed upstream in
`apposed/appose-python` by bumping the pin to `v0.81.0`, and that fix has
since landed on `main`, which now also has a daily workflow to keep its
bundled pixi pin current. **Make sure CellProfiler's `appose` dependency in
`pixi.toml` tracks plain `main`** (`appose = { git =
"https://github.com/apposed/appose-python.git" }`, no branch pin) — do not
pin to an old commit/branch, and if `[pypi-dependencies]`-based plugin
environments ever start failing with the same "unexpected end of file"
symptom again, check this pin first before assuming it's a new bug.

## Existing reference code in CellProfiler-plugins (pre-existing, not ours)

In `CellProfiler-plugins/CP5/active_plugins/`:

- **`cpij/`**: an older, pre-Appose, unrelated `multiprocessing.managers`-based
  ImageJ bridge. Not relevant to this work.
- Earlier Appose proofs-of-concept (`appose_demo.py`, `apposednapari/`, an
  apposified `pixelshuffle.py`) were all reverted after validating the
  pattern — not kept as real plugins. `cellcast.py` (see above) is the
  first one that landed for real.

## Open questions / not yet decided

1. **Service lifecycle / performance**: `run_python_task` currently opens a
   **fresh** `environment.python()` service per call (simplest, safest
   resource semantics). For a plugin processing many image sets in a real
   pipeline run, this means a fresh process (and, for model-based plugins
   like cellcast, a fresh model load) per image, which may be too slow.
   Keeping one persistent `Service` alive for a module instance's lifetime
   would need a CellProfiler module lifecycle hook to close it at the end
   of a run (not investigated — look at whether `Module` has something
   like `post_run`/`on_deactivated`).
2. **Worker glue templating**: the per-plugin glue (convert input `NDArray`
   → numpy → dispatch to worker function(s) → wrap output back into
   `NDArray` → assign to `task.outputs[...]`) is still hand-written per
   plugin, appended as a string onto its `_worker.py` source. `cellcast.py`
   improved this slightly by making the glue a static, runtime-dispatching
   template (see Design decision #3) rather than rebuilding a string per
   call, but it's still duplicated per plugin rather than factored into a
   shared helper in `cellprofiler_core/utilities/appose.py`. Worth doing
   once a third plugin is ported and the shape of the duplication is clearer.
