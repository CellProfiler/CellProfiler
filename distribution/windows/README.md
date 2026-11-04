# Windows packaging

Use the locked Pixi `build` environment and install Inno Setup 6 on the build
machine. Run these PowerShell commands from the repository root:

```powershell
pixi install --locked -e build
Set-Location distribution/windows
pixi run --locked -e build python prepare_java.py build/java-jars
pixi run --locked -e build python -m PyInstaller CellProfiler.spec
```

The executable is written to
`distribution/windows/dist/CellProfiler/CellProfiler.exe`. The release workflow
uses the same working directory.

The spec resolves all four local package roots from the installed packages and
collects Centriole's compiled extensions and Mahotas' vendored DLL directory.
The bundle contains the selected JDK, including its legal notices, under
`_internal/jre`, and the Java jars resolved by `prepare_java.py` under
`_internal/java-jars`. The jar manifest records their hashes and provenance.
Initial Java dependency resolution needs network access or a populated cache.
Frozen execution uses the bundled classpath without resolving Maven endpoints.
`CP_JAVA_HOME` can select another compatible Java runtime.

Before building the installer, set `CP_VERSION` to the CellProfiler version.
The installer copies the already assembled bundle, including its JDK, and does
not add another Java directory. The release workflow runs `ISCC.exe` after
assembling the bundle. For a local installer build, run from
`distribution/windows`:

```powershell
$env:CP_VERSION = pixi run --locked -e build python -m setuptools_scm -c ../../src/frontend/pyproject.toml
pixi run --locked -e build python -c "import os, subprocess; subprocess.run([os.path.join(os.environ['ProgramFiles(x86)'], 'Inno Setup 6', 'ISCC.exe'), 'CellProfiler.iss'], check=True)"
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
