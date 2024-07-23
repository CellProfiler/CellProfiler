{ pkgs, cp_version }:
rec {

  # Cellprofiler
  centrosome = import ./centrosome.nix { inherit pkgs; };
  cellprofiler-library = import ./cellprofiler_library.nix { inherit pkgs centrosome cp_version; };
  jgo = import ./jgo.nix { inherit pkgs; };
  scyjava = import ./scyjava.nix { inherit pkgs jgo; };
  cellprofiler-core = import ./cellprofiler_core.nix { inherit pkgs cp_version centrosome cellprofiler-library scyjava; };
  cellprofiler = import ./cellprofiler.nix { inherit pkgs cp_version cellprofiler-library centrosome cellprofiler-core scyjava; };

}
