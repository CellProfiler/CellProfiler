{
  lib,
  # build deps
  buildPythonPackage,
  fetchFromGitHub,
  pip,
  setuptools,
  # test deps
  pytest,
  cython,
  # runtime deps
  deprecation,
  numpy,
  scipy,
  scikit-image,
}:
buildPythonPackage {
  pname = "centrosome";
  version = "1.3.0";

  src = fetchFromGitHub {
    owner = "afermg";
    repo = "centrosome";
    rev = "87f142083e010c88666efed341a905b98a3977c3";
    sha256 = "sha256-bCbsQ67b67N3+TBOorZJV7URCglHrUj5eLaV648dH/w=";
  };
  pyproject = true;
  buildInputs = [
    cython
    pip
    setuptools
  ];
  propagatedBuildInputs = [
    deprecation
    numpy
    scipy
    scikit-image
    pytest
  ];
  pythonImportsCheck = [];

  meta = {
    description = "Centrosome";
    homepage = "https://cellprofiler.org";
    license = lib.licenses.bsd3;
  };
}
