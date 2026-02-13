# Linux Distribution

Linux distributions of CellProfiler are done via Flatpak.

## Flatpak Manifest

The [manifest](https://docs.flatpak.org/en/latest/flatpak-builder-command-reference.html#flatpak-manifest) file, `org.cellprofiler.CellProfiler.yml`, contains the instructions to build the Flatpak with `flatpak-builder`. It is fairly typical other than the usage of `pixi` explained below. Refer to the [Flatpak docs](https://docs.flatpak.org/en/latest/introduction.html) to understand its contents.

The command to build the package is:

```
flatpak-builder --force-clean --user --install-deps-from=flathub --repo=repo --install builddir org.cellprofiler.Cellprofiler.yml
```

## Pixi Pack

CellProfiler builds are managed with `pixi` on all platforms, including Linux. This allows us to source from conda-forge along with PyPI, set enviornment variables, activation scripts, etc. It also allows us to use the `pixi.lock` file for deterministic environments. Flatpak has a limited amount of build systems (e.g. `autotools`, `cmake`, `meson`, etc.). These are robust, but not what CellProfiler uses. In order to build Python applications, Flatpak [recommends](https://docs.flatpak.org/en/latest/python.html) using the plain `simple` build system, and using the `pip` bundled with the Flatpak runtime and sdk (`org.freedesktop.Platform` and `org.freedesktop.Sdk`). They also recommend including the full dependency package manifest as Flatpak module sources, rather than attempting to do a `pip install` from within the build environment. Doing so avoids two things 1) performing non-determenistic dependency resolution via `pip` (since it doesn't use lock files by default) and 2) network calls from `pip` to PyPI from within the build environment. Instead exact urls to exact versions of PyPI packages are given as sources, and they are retrieved as inputs to the build. Using `pip` isn't strictly problematic for building CellProfiler, however the point of using `pixi` is to allow for extra packages not on PyPI, reproducible environments via lock file, etc. `pip` is insufficient for installing the `jdk` and setting the `JAVA_HOME` environment variable for instance.

There are at least two ways of using `pixi` to build CellProfiler with `flatpak-builder`. The first is to enable network access in the module `build-args` (`--share=network`), use `curl` to install `pixi`, build the `pixi` environment with `pixi install` (which will download the packages), copy the packages to the Flatpak runtime `/app` folder, use `pixi shell-hook` to generate an activation script, and use that to invoke the `cellprofiler` package to launch the GUI. An example of that was done successfully in commit history (`git log` and search for the commit preceding "[linux] Refactor manifest without build-time network access"). Enabling network access is [not allowed](https://github.com/flathub/flathub/issues/3392) when deploying to Flathub (i.e. using the `flathub-builder` package for building). Therefore the second approach is to use `pixi` outside of the Flatpak build environment, package the environment with [pixi-pack](https://pixi.prefix.dev/latest/deployment/pixi_pack/) as a self-extracting binary, specify that binary as a module source of the Flatpak build environment, unpackage it in the Flatpak runtime `/app` folder, and use `pip` with network access disabled to install CellProfiler itself. The caveat to using `pixi-pack` is that currently it can only use pre-built conda packages and wheels. Source distributions from PyPI are not yet supported, although there are open PRs addressing that:
* Open `pixi-pack` [PR](https://github.com/Quantco/pixi-pack/pull/259) to allow local wheels.
* Open `pixi-pack` [PR](https://github.com/Quantco/pixi-pack/pull/244) to build packages from source conda packages.

At the time of writing, the `pixi.toml` targeting the linux-focused publish/production environment is carefully curated to avoid all source distributions from PyPI. Before running the Flatpak build process, this command should produce zero results:

```
pixi list --environment prod --platform linux-64 --fields name,version,build,size,kind,source,url | grep pythonhosted | grep tar
```

Once that is confirmed, the command to pack the environment is:

```
pixi exec pixi-pack --environment prod --platform linux-64 --use-cache ~/.pixi_pack/cache --ignore-pypi-non-wheel --create-executable pixi.toml
```

That will generate an `environment.sh` file which should be placed in `distribution/linux` prior to building the Flatpak.


## Why not pyinstaller?

Currently there is no `pyinstaller`-based build targeting Linux. There is in theory no reason it shouldn't work but simply hasn't been done. It may even be easeir to use `pyinstaller` over Flatpak since Flatpak has a number of limitations for the base runtime, and tooling available at build time. As explained above, for Python apps in particular, `flatpak-builder` only has out of the box support for `pip`, and not for more modern tools such as `uv` or `pixi`.
For good and for bad, Flatpak's sandboxing also imposes restrictions that `pyinstaller` does not. This means the user can restrict things like network access very easily using Flatpak tools (e.g. Flatseal), but means we have to use special Linux-specific APIs ([XDG Desktop Portal](https://flatpak.github.io/xdg-desktop-portal/)) for various things like accessing the file system. The benefit of Flatpak over `pyinstaller` is the potential for space savings. With `pyinstaller`, all dependencies (including the many shared libraries / binaries CellProfiler relies on) other than `glibc` are packaged into the bundle. With Flatpak, the base runtime is shared with other Flatpak apps, reducing some space overhead. Flatpak is also more standardized, and Flatpak apps can be published on common repositories such as Flathub.


