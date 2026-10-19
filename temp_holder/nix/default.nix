{
  lib,
  pkgs,
  python3Packages,
  cp_version ? "master",
}: let
  callPackage = lib.callPackageWith (pkgs // packages // python3Packages);
  packages = {
    centrosome = callPackage ./centrosome.nix {};
    jgo = callPackage ./jgo.nix {};
    scyjava = callPackage ./scyjava.nix {};
    cellprofiler-library = callPackage ./cellprofiler_library.nix {inherit cp_version;};
    cellprofiler-core = callPackage ./cellprofiler_core.nix {inherit cp_version;};
    cellprofiler = callPackage ./cellprofiler.nix {inherit cp_version;};
  };
in
  packages
