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
    rev = "4eff192965db7815755b5823d5eeb8fc63303b81";
    sha256 = "sha256-F9bVzj0e3VhPOkBPlV+IxLbT9hBcqJiXlnbnxEckrtM=";
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
  pythonImportsCheck = [ ];

  meta = {
    description = "Centrosome";
    homepage = "https://cellprofiler.org";
    license = lib.licenses.bsd3;
  };
}
