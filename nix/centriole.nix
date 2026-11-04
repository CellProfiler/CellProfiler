{
  lib,
  # build deps
  buildPythonPackage,
  fetchPypi,
  setuptools-scm,
  cython,
  # test deps
  pytest,
  # runtime deps
  deprecation,
  contourpy,
  pillow,
  pywavelets,
  numpy,
  scipy,
  scikit-image,
  cp_version,
  python3Packages,
}:
let
  helper = import ./version_patch.nix { inherit lib python3Packages; };
in
buildPythonPackage rec {
  pname = "centriole";
  version = cp_version;

  src = ../src/subpackages/centriole;
  pyproject = true;

  # Relax deps constraints
  postPatch = ''
    substituteInPlace pyproject.toml \
      ${(helper.patchPackageVersions ../src/subpackages/centriole/pyproject.toml).subStr}
    echo 'fallback_version = "${version}"' >> pyproject.toml
  '';

  buildInputs = [
    pytest
    cython
    numpy
    (setuptools-scm.overrideAttrs rec {
      version = "8.1.0";
      src = fetchPypi {
        pname = "setuptools_scm";
        inherit version;
        hash = "sha256-Qt6htldxy6k7elFdZaZdgkblYHaKZrkQalksjn8myKc=";
      };
    })
  ];

  propagatedBuildInputs = [
    deprecation
    contourpy
    pillow
    pywavelets
    numpy
    scipy
    scikit-image
  ];

  pythonImportsCheck = [ "centriole" ];

  meta = {
    description = "Centriole";
    homepage = "https://github.com/CellProfiler/CellProfiler";
    license = lib.licenses.bsd3;
  };
}
