{ pkgs, centrosome, cp_version, cellprofiler-library, cellprofiler-core, scyjava }:

pkgs.python3Packages.buildPythonPackage rec {
  pname = "cellprofiler";
  version = cp_version;

  src = ../src/frontend;
  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace "docutils==0.15.2" "docutils>=0.15.2" \
      --replace "scipy>=1.4.1,<1.11" "scipy>=1.4.1" \
      --replace "centrosome~=1.2.2" "centrosome~=1.2.3" \
      --replace "wxPython==4.2.0" "wxPython~=4.2"
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
    cellprofiler-core
    centrosome
    boto3
    docutils
    imageio
    inflect
    jinja2
    joblib
    mahotas
    mysqlclient
    pillow
    sentry-sdk
    requests
    future
    fsspec
    h5py
    lxml
    matplotlib
    numpy
    psutil
    pyzmq
    scikit-image
    scikit-learn
    scipy
    scyjava
    six
    tifffile
    wxPython_4_2
    rapidfuzz
    packaging
  ];

  pythonImportsCheck = [];

  meta = with pkgs.lib; {
    description = "Cellprofiler core";
    homepage = "https://cellprofiler.org";
    # license = license.mit;
  };
}
