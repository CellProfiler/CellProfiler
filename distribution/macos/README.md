# macOS packaging

Build with the repository's Pixi `build` environment on the target Mac architecture:

```sh
pixi install -e build
make -C distribution/macos
```

The application is written to `distribution/macos/dist/CellProfiler.app`. The build uses the current source tree and installed package version. It does not check out a branch or install packages with pip. Java 11 and MySQL client libraries come from Pixi. The recipe does not invoke Homebrew or UPX. PyInstaller also collects native dependencies of installed wheels; inspect its dependency report for any libraries supplied outside Pixi.

`prepare_java.py` resolves CellProfiler's Bio-Formats dependencies during the build, verifies the reader class, and copies the resolved jars and their hashes into the build directory. Initial dependency resolution needs network access or a populated Java dependency cache. Maven endpoints are currently unversioned; keep the generated jar manifest with each release artifact to record the selected versions and hashes. The application includes those jars, the JDK and its legal notices, and the MySQL client authentication plugins. It locates these resources relative to the application at runtime. `CP_JAVA_HOME` can override the bundled Java runtime.

Build and output paths can be changed without editing the source tree:

```sh
make -C distribution/macos CP_BUILD="/absolute/path/build" CP_OUTPUT="/absolute/path/output"
```

To use an installed Developer ID signing certificate, set `CP_CODESIGN_IDENTITY` before building. PyInstaller signs the native files and application while assembling the bundle. Without that variable, PyInstaller uses ad-hoc signing for local testing.

```sh
CP_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" make -C distribution/macos
sh distribution/macos/osx-codesign.sh distribution/macos/dist/CellProfiler.app
```

Signature verification does not establish notarization or Gatekeeper acceptance. Notarization and testing a quarantined downloaded artifact require separate release checks.

Before distributing, copy the complete `.app` to an unrelated path containing spaces. Test direct execution through `Contents/MacOS/cp` and Finder launch, image loading, a representative pipeline, Java image reading, native processing, workers and normal shutdown. Remove development environment variables and make the original source/build environment unavailable. Check loaded libraries and symlinks for dependencies outside the bundle, allowing macOS system libraries and explicitly configured plugin environments. Appose plugins use their own managed environments; the application bundle is not a general Python interpreter for them.

Local relocation tests do not replace testing on a clean Mac. Record the tested macOS version, architecture, interpreter, artifact hashes, loaded libraries and scientific output comparisons. Repeat for each distributed architecture and the oldest supported macOS release.

Use `make -C distribution/macos clean` to remove the selected build and output directories.
