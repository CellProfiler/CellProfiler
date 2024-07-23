{ pkgs }:
rec {

  # Cellprofiler
  centrosome = import ./centrosome.nix { inherit pkgs; };
  cellprofiler-library = import ./cellprofiler_library.nix { inherit pkgs centrosome; };
  jgo = import ./jgo.nix { inherit pkgs; };
  scyjava = import ./scyjava.nix { inherit pkgs jgo; };
  cellprofiler-core = import ./cellprofiler_core.nix { inherit pkgs centrosome cellprofiler-library scyjava; };
  cellprofiler = import ./cellprofiler.nix { inherit pkgs cellprofiler-library centrosome cellprofiler-core scyjava; };

}
