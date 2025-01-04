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
  python3Packages,
}:
let
  helper = import ./version_patch.nix { inherit lib python3Packages; };
in
buildPythonPackage rec {
  pname = "cellprofiler";
  version = cp_version;

  src = ../src/frontend;
  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml \
      ${(helper.patchPackageVersions ../src/frontend/pyproject.toml).subStr}
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

  nativeBuildInputs = [ git ];
  buildInputs = [
    pytest
    setuptools
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
