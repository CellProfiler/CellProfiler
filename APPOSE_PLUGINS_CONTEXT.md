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
   similar) task input and dispatch on it at runtime. The glue string is
   then computed once at import time instead of being rebuilt per `run()`
   call. As of `RunCellcast`'s current form, the dispatch `if`/`elif`
   itself lives in `_worker.py` (a `get_model_scaffold(...)` function
   returning a `{"cache_key", "init", "predict_labels"}` dict), while the
   epilogue appended in `cellcast.py` is fully generic Appose plumbing
   (NDArray marshaling + the `_get_cached_model` caching helper, see
   "Persistent services" below) that calls whatever scaffold dict
   `_worker.py` hands back. See `cellcast.py`/`cellcast/_worker.py` for the
   pattern — this generic epilogue shape is a good candidate to eventually
   factor out per open question 1 below.

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
- `<name>/_worker.py`: pure functions (plus, per the current `RunCellcast`
  shape, the per-case dispatch itself — see design decision #3), zero
  `cellprofiler_core` imports. This file is read as **plain text** by the
  frontend `.py` file and concatenated with a small inline "glue" epilogue
  that converts the raw input `appose.NDArray` to a numpy array, calls into
  `_worker.py`'s dispatch, wraps the result back into a fresh `NDArray`,
  and assigns it to `task.outputs["<key>"]`.

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
- `get_service(env_spec)` / `close_service(env_spec)` start (or reuse a
  cached) **persistent** `environment.python()` service — i.e. a worker
  process that stays alive across many calls, instead of one per call. See
  "Persistent services" below.
- `run_python_task(script, inputs=None, environment=None, service=None)`
  copies any `numpy.ndarray` inputs into Appose shared memory, runs `script`
  against either a fresh `environment.python()` service (pass
  `environment`) or a reused one (pass `service`), copies any
  `appose.NDArray` outputs back into plain numpy arrays, and releases all
  shared memory before returning.
- All of the above only use the oldest-common-denominator Appose `NDArray`
  API (manual `appose.NDArray(dtype=..., shape=...)` construct +
  `.ndarray()`) — see Gotcha 1.

## Persistent services (performance)

A fresh service per call is simplest but pays for a new worker process —
and, for model-based plugins, a fresh model load — on every image set.
`get_service()`/`close_service()` plus `run_python_task(..., service=...)`
let a plugin reuse one long-lived worker process across calls instead.

Reusing the *process* is automatic once you pass `service=` instead of
`environment=`. Reusing an expensive *resource inside* that process (e.g. a
loaded model) is not automatic — each task's script starts with a fresh
global namespace containing only `task` plus whatever a previous task
explicitly persisted via `task.export(name=value)`. `RunCellcast`'s glue
(`cellcast.py`'s `_get_cached_model` helper) shows the pattern: compute a
cache key from whatever settings actually affect model construction, check
it against `globals().get("_cellcast_cache_key")`, and only rebuild +
re-export when it differs. This way a settings change (including one made
interactively in GUI test/debug mode, which reruns the same module instance
and reuses the same service) is still picked up correctly instead of
silently serving stale results. Measured effect for `RunCellcast`: first
call ~0.57s (service start + model build), subsequent calls with an
unchanged cache key ~0.03s.

Lifecycle, across CellProfiler's three execution modes (see
`cellprofiler_core/module/_module.py` for the hooks referenced below):

- **Headless** and **GUI run mode's main process**: `Module.prepare_run`/
  `post_run` run once per whole pipeline run, so `post_run` is a correct
  place to call `close_service()`. `RunCellcast.post_run` does this.
- **GUI run mode's worker subprocesses** (where `Module.run` — and
  therefore the plugin's actual service — lives): workers are long-lived
  OS subprocesses that process many jobs over the analysis's lifetime,
  reusing the same deserialized `Module` instance throughout, which is
  exactly what makes caching worthwhile here. But `prepare_run`/`post_run`
  are **never called inside a worker process** — only in the GUI's main
  process, which holds a *different* `Module` instance. There is no
  teardown hook reachable from worker-local module code at all, so cleanup
  here relies entirely on `get_service()`'s `atexit.register(close_service,
  env_spec)` firing when the worker process itself exits (normal end of
  analysis, or a crash) — see Gotcha 5 for why `close_service()` has to be
  a *bounded*, tree-aware kill rather than a graceful `service.close()`.
- **GUI test/debug mode**: never calls `prepare_run`/`post_run` either, and
  reruns the same module instance (so the same cached service) arbitrarily,
  including after settings edits. Caching still works and self-invalidates
  correctly (same mechanism as above), but nothing ever explicitly calls
  `close_service()` in this mode until the process exits — same
  atexit-only cleanup story as the worker case above, and the same reason
  Gotcha 5's bound matters here too (this is literally the mode where the
  original hang-on-quit bug was discovered).

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

### 5. A persistent `Service`'s own threads can deadlock CellProfiler's quit sequence — `atexit` alone cannot fix this

Symptom: quitting the CellProfiler GUI (after using a persistent-service
plugin like `RunCellcast`) hung indefinitely until force-killed from the
shell. A first fix attempt — making `close_service()` bounded and
process-tree-aware (see below) and registering it via
`atexit.register(close_service, env_spec)` in `get_service()` — did not
resolve it. Root cause has two parts, confirmed independently of
CellProfiler via a standalone, Appose-only MCVE (see below) before being
fixed here.

**Part 1 (the actual hang, dominant cause): `atexit` runs too late to help.**
Appose's `Service` runs its stdout/stderr/monitor plumbing on plain,
non-daemon `threading.Thread`s (`appose/service.py`'s `start()`). CPython's
interpreter shutdown (`Py_FinalizeEx`) joins **all non-daemon threads**
(`wait_for_thread_shutdown()`) **before** running `atexit` callbacks
(`call_py_exitfuncs()`) — confirmed empirically with a minimal, non-Appose
script. Appose's reader threads only return once the worker's pipes hit
EOF, i.e. once something has told the worker to exit. If that "something"
is an `atexit` callback (the standard, idiomatic place to put this kind of
cleanup — and exactly what `get_service()` does), it **never gets
scheduled**, because the thread-join step it would need to unblock is
already stuck, waiting for it. Any caller that registers `atexit.register`
to close a still-open `Service` hits this — it is not specific to
`PixiBuilder`, pixi, or cellcast. The result is a permanent hang with no
exception and no log output, as opposed to a bounded delay.

**Part 2 (compounding, only matters once something does call `close()`):**
For a `PixiBuilder`-built environment, the worker is launched as `pixi run
--manifest-path ... python -c "...python_worker..."` — Appose's
`Service._process` is the **`pixi` wrapper**, not the actual
`python_worker` process `pixi` spawns as its child. Confirmed via `psutil`:
calling `Service.kill()` only kills the wrapper; the real worker survives
as an orphan (reparented to pid 1) and keeps running until it finishes
whatever task it was mid-execution on (if ever) and only then notices
stdin EOF.

**Fix applied**, in `cellprofiler_core/utilities/appose.py`:

- `get_service()` now starts the service via a new `_start_as_daemon()`
  helper instead of calling `service.start()` directly: it runs
  `service.start()` from inside a short-lived daemon thread. Since
  `threading.Thread(daemon=None)` (Appose's default - it never passes
  `daemon=`) inherits the daemon flag of *whichever thread constructs it*,
  every thread Appose creates transitively - its three I/O threads included
  - comes out daemon too. Daemon threads are **abandoned, not joined**, at
  interpreter exit, so the Part 1 deadlock is now structurally impossible,
  regardless of whether anything ever calls `close()`/whether `atexit` gets
  a chance to run. This is the real fix for the hang itself.
- `close_service()` is still bounded and process-tree-aware (closes
  gracefully, waits up to `_CLOSE_TIMEOUT_SECONDS` = 5s, then force-kills
  via `_kill_process_tree()`, which uses `psutil` — already a
  `cellprofiler-core` dependency — to enumerate and kill `Service._process`
  and all its descendants, not just the immediate child). This remains
  worthwhile purely to avoid *leaking* the worker subprocess (Part 2) when
  a service is never explicitly closed — and now that Part 1's deadlock is
  gone, `atexit`-triggered calls to it actually get to run.

Verified together: a service with a 30s task in flight, cleaned up via
`atexit` only (no explicit `close()`/`post_run` call anywhere in the test),
exits with code 0 in ~8s total, with zero surviving processes afterward
(checked via `ps aux | grep python_worker` immediately after exit) — versus
hanging forever (confirmed via `timeout`, process never exited) before this
fix.

**Standalone MCVE** (no CellProfiler involved): a bare pixi project
(`python` + `appose` + `psutil`, all conda-forge) at
`appose-hang-mcve/{pixi.toml,repro.py}` (built during this session; ask for
its current location/to have it re-created if it's no longer on disk - it
lived under a session scratch directory) with five `pixi run
scenario-{a,b,c,d,e}` targets:
- **a**: open a service, run one task, return without closing it →
  process hangs forever at exit (`timeout` has to kill it); the real worker
  survives untouched as an orphan unless manually killed afterward.
- **b**: a long task in flight, shut down using only Appose's *documented*
  public API (`close()`, wait, `kill()`) → two compounding problems:
  `kill()` only signals the `pixi` wrapper, never the real worker; and even
  though `close()` *does* reach the real worker (same inherited stdin pipe)
  and its read loop returns promptly, the worker process still can't exit
  until the in-flight task finishes, because the task runs on its own
  non-daemon thread (`python_worker.py`'s per-task thread) that the
  worker's own interpreter shutdown must join first - the same root cause
  as scenario c, one level deeper. `wait_for()` doesn't return until
  close to the task's full duration, not shortly after `kill()`.
- **c**: same long task, cleaned up via `atexit.register` only (the
  idiomatic pattern) → hangs forever; the atexit callback's own print
  statement never fires, proving `atexit` never ran at all - the one that
  actually matters for the hang this session chased down.
- **d**: identical to **c**, except `service.start()` is called from a
  daemon thread first → exits in ~1s, `atexit` callback fires correctly
  (confirms the fix we applied in `_start_as_daemon()`).
- **e**: identical to **b**, but after `close()` the real worker PID (found
  via `psutil`, not any public API) is SIGKILL-ed directly instead of/in
  addition to `service.kill()` → `wait_for()` returns in well under a
  second regardless of remaining task duration, confirming scenario b's
  delay is attributable entirely to the real worker process, not the
  wrapper.

Filed upstream: <https://github.com/apposed/appose/issues/37>.

## Existing reference code in CellProfiler-plugins (pre-existing, not ours)

In `CellProfiler-plugins/CP5/active_plugins/`:

- **`cpij/`**: an older, pre-Appose, unrelated `multiprocessing.managers`-based
  ImageJ bridge. Not relevant to this work.
- Earlier Appose proofs-of-concept (`appose_demo.py`, `apposednapari/`, an
  apposified `pixelshuffle.py`) were all reverted after validating the
  pattern — not kept as real plugins. `cellcast.py` (see above) is the
  first one that landed for real.

## Open questions / not yet decided

1. **Worker glue templating**: the per-plugin glue (convert input `NDArray`
   → numpy → dispatch to worker function(s) → wrap output back into
   `NDArray` → assign to `task.outputs[...]`) is still hand-written per
   plugin, appended as a string onto its `_worker.py` source. `cellcast.py`
   improved this slightly by making the glue a static, runtime-dispatching
   template (see Design decision #3) rather than rebuilding a string per
   call, but it's still duplicated per plugin rather than factored into a
   shared helper in `cellprofiler_core/utilities/appose.py`. Worth doing
   once a third plugin is ported and the shape of the duplication is clearer.
