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
    rev = "15093d60";
    sha256 = "";
  };
  pyproject = true;
  buildInputs = [
    pytest
    cython
    pip
    setuptools
  ];
  propagatedBuildInputs = [
    deprecation
    numpy
    scipy
    scikit-image
  ];
  pythonImportsCheck = [];

  meta = {
    description = "Centrosome";
    homepage = "https://cellprofiler.org";
    license = lib.licenses.bsd3;
  };
}
