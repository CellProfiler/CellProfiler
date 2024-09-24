{
  lib,
  # build deps
  buildPythonPackage,
  fetchPypi,
  setuptools,
  setuptools-scm,
  git,
  # test deps
  pytest,
  # runtime deps
  cellprofiler-library,
  cellprofiler-core,
  centrosome,
  imageio,
  inflect,
  jinja2,
  joblib,
  mahotas,
  mysqlclient,
  pillow,
  sentry-sdk,
  requests,
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
  packaging,
  scikit-learn,
  six,
  tifffile,
  wxPython_4_2,
  rapidfuzz,
  cp_version,
}:
buildPythonPackage rec {
  pname = "cellprofiler";
  version = cp_version;

  src = ../src/frontend;
  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace "docutils==0.15.2" "docutils>=0.15.2" \
      --replace "wxPython~=4.2.0" "wxPython~=4.2" \
      --replace-warn "h5py~=3.6.0" "h5py<=3.11.0" \
      --replace-warn "numpy~=1.24.4" "numpy<=1.26.4" \
      --replace-warn "scikit-image~=0.20.0" "scikit-image<=0.22.0" \
      --replace-warn "scipy>=1.9.1,<1.11" "scipy>=1.9.1,<=1.13" \
      --replace-warn "pyzmq~=22.3.0" "pyzmq<=25.1.2" \
      --replace-warn "boto3~=1.28.41" "boto3<=1.34.58" \
      --replace-warn "imageio~=2.31.3" "imageio<=2.34.1" \
      --replace-warn "inflect~=7.0.0" "inflect<=7.2.0" \
      --replace-warn "joblib~=1.3.2" "joblib<=1.4.0" \
      --replace-warn "Pillow~=10.0.0" "Pillow<=10.3.0" \
      --replace-warn "sentry-sdk>=0.18.0,<=1.31.0" "sentry-sdk>=0.18.0,<=1.45.0" \
      --replace-warn "scikit-learn~=1.3.0" "scikit-learn<=1.4.2" \
      --replace-warn "tifffile>=2022.4.8,<2022.4.22" "tifffile>=2022.4.8,<=2024.4.18" \
      --replace-warn "rapidfuzz~=3.0.0" "rapidfuzz<=3.9.1"
    echo 'fallback_version = "${version}"' >> pyproject.toml
  '';

  preBuild = ''
    git init .
    git config user.name 'Anonymous'
    git config user.email '<>'
    git add .
    git commit -m "hack: please fix me later"
    git tag "v${version}"
  '';

  nativeBuildInputs = [git];
  buildInputs = [
    pytest
    setuptools
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

  pythonImportsCheck = [
    "cellprofiler"
  ];

  meta = {
    description = "Cellprofiler core";
    homepage = "https://github.com/CellProfiler/CellProfiler";
    license = lib.licenses.bsd3;
  };
}
