{ pkgs, cp_version, centrosome, cellprofiler-library, scyjava }:

pkgs.python3Packages.buildPythonPackage rec {
  pname = "cellprofiler_core";
  version = cp_version;

  src = ../src/subpackages/core;
  pyproject = true;  

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace "docutils==0.15.2" "docutils>=0.15.2" \
      --replace "scipy>=1.4.1,<1.11" "scipy>=1.4.1" \
      --replace "centrosome~=1.2.2" "centrosome~=1.2.3"
    echo 'fallback_version = "${version}"' >> pyproject.toml
  '';

  buildInputs  = with pkgs.python3Packages; [
    pytest
    (
      setuptools-scm.overrideAttrs rec {
        version = "8.1.0";
        src = pkgs.fetchPypi {
          pname = "setuptools_scm";
          inherit version;
          hash = "sha256-Qt6htldxy6k7elFdZaZdgkblYHaKZrkQalksjn8myKc=";
        };

      }
    )
  ];

  propagatedBuildInputs = with pkgs.python3Packages; [
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

  pythonImportsCheck = [];

  meta = with pkgs.lib; {
    description = "Cellprofiler core";
    homepage = "https://cellprofiler.org";
    # license = license.mit;
  };
}
