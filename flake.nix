{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    systems,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
        };
        
      in
        with pkgs; rec {
          callPackage = lib.callPackageWith (pkgs // mypackages // python3Packages);
          
          mypackages = {
              centrosome = callPackage ./centrosome.nix {};
            };

          # formatter = pkgs.alejandra;
          devShells = {
            default = let
              python_with_pkgs = (pkgs.python3.withPackages(pp: [
                mypackages.centrosome
                ]));
              in
              mkShell {
                NIX_LD = runCommand "ld.so" {} ''
                  ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
                '';
                NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
                  # Add needed packages here
                  stdenv.cc.cc
                  libGL
                  zlib
                  glib
                ];
                packages = [
                  python_with_pkgs
                  # git
                  # gtk3
                  glib
                  pkg-config
                ];
                venvDir = "./.venv";
                postVenvCreation = ''
                  unset SOURCE_DATE_EPOCH
                '';
                postShellHook = ''
                  unset SOURCE_DATE_EPOCH
                '';
                shellHook = ''
                  export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
                  runHook venvShellHook
                  export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
                '';
              };
          };
        }
    );
}
