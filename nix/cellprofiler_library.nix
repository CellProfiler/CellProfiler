{
  lib,
  # build deps
  buildPythonPackage,
  fetchPypi,
  setuptools-scm,
  # test deps
  pytest,
  # runtime deps
  centrosome,
  matplotlib,
  numpy,
  mahotas,
  scikit-image,
  scipy,
  packaging,
  cp_version,
  python3Packages,
}:
let
  helper = import ./version_patch.nix { inherit lib python3Packages; };
in
buildPythonPackage rec {
  pname = "cellprofiler_library";
  version = cp_version;

  src = ../src/subpackages/library;
  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml \
      ${(helper.patchPackageVersions ../src/subpackages/library/pyproject.toml).subStr}
    echo 'fallback_version = "${version}"' >> pyproject.toml
  '';

  buildInputs = [
    pytest
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
    numpy
    scikit-image
    scipy
    mahotas
    centrosome
    matplotlib
    packaging
  ];

  pythonImportsCheck = [ "cellprofiler_library" ];

  meta = {
    description = "Cellprofiler library";
    homepage = "https://github.com/CellProfiler/CellProfiler";
    license = lib.licenses.bsd3;
  };
}
