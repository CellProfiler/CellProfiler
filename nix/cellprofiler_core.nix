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
}:
buildPythonPackage rec {
  pname = "cellprofiler_core";
  version = cp_version;

  src = ../src/subpackages/core;
  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-warn "docutils==0.15.2" "docutils>=0.15.2" \
      --replace-warn "h5py~=3.6.0" "h5py<=3.11.0" \
      --replace-warn "numpy~=1.24.4" "numpy<=1.26.4" \
      --replace-warn "scikit-image~=0.20.0" "scikit-image<=0.22.0" \
      --replace-warn "scipy>=1.9.1,<1.11" "scipy>=1.9.1,<=1.13" \
      --replace-warn "zarr~=2.16.1" "scikit-image<=2.17.2" \
      --replace-warn "pyzmq~=22.3.0" "pyzmq<=25.1.2" \
      --replace-warn "google-cloud-storage~=2.10.0" "google-cloud-storage<=2.16.0"
    echo 'fallback_version = "${version}"' >> pyproject.toml
  '';

  buildInputs = [
    pytest
    (
      setuptools-scm.overrideAttrs rec {
        version = "8.1.0";
        src = fetchPypi {
          pname = "setuptools_scm";
          inherit version;
          hash = "sha256-Qt6htldxy6k7elFdZaZdgkblYHaKZrkQalksjn8myKc=";
        };
      }
    )
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

  pythonImportsCheck = ["cellprofiler_core"];

  meta = {
    description = "Cellprofiler core";
    homepage = "https://github.com/CellProfiler/CellProfiler";
    license = lib.licenses.bsd3;
  };
}
