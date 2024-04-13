# Windows

Make sure the build machine has `pywin32` and `InnoSetup` installed.

To Run:

* run `pyinstaller CellProfiler.spec` 
* Make sure `JDK_HOME` and `CP_VERSION` environment variables are set
    * Alternatively just open the `CellProfiler.iss` file and manually set `JDKPath` and `MyAppVersion` at the top
    * `CP_VERSION` should be the value returned from (run from this directory): `python -m setuptools_scm --strip-dev -c ../../frontend/pyproject.toml`
    * exclude `--strip-dev` to include the dev/local parts of the version string
* Run `InnoSetup` on `.iss` file