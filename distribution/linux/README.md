# Linux Distribution

Linux distributions of CellProfiler are done via Flatpak.

## Pixi Pack

Open `pixi-pack` [PR](https://github.com/Quantco/pixi-pack/pull/259) to allow local wheels.
Open `pixi-pack` [PR](https://github.com/Quantco/pixi-pack/pull/244) to build packages from source conda packages.

`pixi-pack --environment prod --platform linux-64 --use-cache ~/.pixi_pack/cache --ignore-pypi-non-wheel --create-executable pixi.toml`

`pixi list --environment prod --platform linux-64 --fields name,version,build,size,kind,source,url | grep pythonhosted | grep tar`

## Why not pyinstaller?

Currently there is no `pyinstaller`-based build targeting Linux. There is in theory no reason it shouldn't work but simply hasn't been done.
It may even be easeir to use `pyinstaller` over Flatpak since Flatpak has a number of limitations for the base runtime, and tooling available at build time.
For Python apps in particular, `flatpak-builder` only has out of the box support for `pip`, and not for more modern tools such as `uv` or `pixi`.
For good and for bad, Flatpak's sandboxing also imposes restrictions that `pyinstaller` does not.
This means the user can restrict things like network access very easily using Flatpak tools (e.g. Flatseal), but means we have to use special Linux-specific APIs (XDG Desktop Portal) for things like network access.
The benefit of Flatpak over `pyinstaller` is the potential for space savings.
With `pyinstaller`, all dependencies (including the many shared libraries / binaries CellProfiler relies on) other than `glibc` are packaged into the bundle.
With Flatpak, the base runtime is shared with other Flatpak apps, reducing some space overhead.
Flatpak is also more standardized, and Flatpak apps can be published on common repositories such as Flathub.


