{
  lib,
  # build deps
  buildPythonPackage,
  fetchPypi,
  setuptools-scm,
  # test deps
  pytest,
  # runtime deps
  cellprofiler-library,
  centrosome,
  boto3,
  docutils,
  future,
  fsspec,
  h5py,
  lxml,
  matplotlib,
  numpy,
  psutil,
  pyzmq,
  scikit-image,
  scipy,
  scyjava,
  zarr,
  google-cloud-storage,
  packaging,
  cp_version,
  python3Packages,
}:
let
  helper = import ./version_patch.nix { inherit lib python3Packages; };
in
buildPythonPackage rec {
  pname = "cellprofiler_core";
  version = cp_version;

  src = ../src/subpackages/core;
  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml \
      ${(helper.patchPackageVersions ../src/subpackages/core/pyproject.toml).subStr}
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
    cellprofiler-library
    centrosome
    boto3
    docutils
    future
    fsspec
    h5py
    lxml
    matplotlib
    numpy
    psutil
    pyzmq
    scikit-image
    scipy
    scyjava
    zarr
    google-cloud-storage
    packaging
  ];

  pythonImportsCheck = [ "cellprofiler_core" ];

  meta = {
    description = "Cellprofiler core";
    homepage = "https://github.com/CellProfiler/CellProfiler";
    license = lib.licenses.bsd3;
  };
}
