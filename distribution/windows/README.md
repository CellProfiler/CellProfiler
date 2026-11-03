# Windows packaging

Use the locked Pixi `build` environment and install Inno Setup 6 on the build
machine. Run these PowerShell commands from the repository root:

```powershell
pixi install --locked -e build
pixi install --locked -e ci-tools
Set-Location distribution/windows
pixi run --locked -e build python -m PyInstaller CellProfiler.spec
pixi run --locked -e ci-tools python check_frozen.py dist/CellProfiler/CellProfiler.exe
```

The executable is written to
`distribution/windows/dist/CellProfiler/CellProfiler.exe`. The release workflow
uses the same working directory.

The spec resolves all four local package roots from the installed packages and
collects Centriole's compiled extensions. Do not treat a successful build as a
successful executable check. `check_frozen.py` runs `--help` from a temporary
directory with developer Python, Pixi and Java activation paths removed. It
returns the executable's exit status, or 124 if startup times out.

Before building the installer, set `JDK_HOME` to the activated build environment's
`JAVA_HOME` and `CP_VERSION` to its CellProfiler version. The release workflow
sets these values and runs `ISCC.exe` on `CellProfiler.iss` after checking the
executable. For a local installer build, run from `distribution/windows`:

```powershell
$env:CP_VERSION = pixi run --locked -e build python -m setuptools_scm -c ../../src/frontend/pyproject.toml
pixi run --locked -e build python -c "import os, subprocess; os.environ['JDK_HOME'] = os.environ['JAVA_HOME']; subprocess.run([os.path.join(os.environ['ProgramFiles(x86)'], 'Inno Setup 6', 'ISCC.exe'), 'CellProfiler.iss'], check=True)"
```

For both `dev` and `build`, verify that `JAVA_HOME` expands into the active Pixi
environment and contains `bin/server/jvm.dll`. Check `java -version`,
`mvn --version`, and an actual JPype JVM start/call/shutdown without overriding
`JAVA_HOME`. Check `prod` activation too. The `profile` environment supports Linux
and macOS only.

Use fresh environments for acceptance. If reusing an environment upgraded from
Python 3.9, audit installed wheel tags; the Windows migration probe found stale
JPype and pyzmq wheels despite a locked install. Reinstall only affected packages
with `pixi reinstall --locked -e dev jpype1` or
`pixi reinstall --locked -e dev pyzmq` if that problem is present.

For release acceptance, copy the bundle outside the source/build directory and
make the original source tree unavailable. Run a headless TIFF pipeline, real GUI
startup/rendering/shutdown, parallel analysis workers, Java image reading, and
database authentication/plugin checks. Test the installer on a clean machine.
The startup check alone does not establish Java/resource closure or relocation.
