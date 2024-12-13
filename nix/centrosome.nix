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
    rev = "d9313e13c557264f8899f6bac3a5210e4580b40e";
    sha256 = "sha256-ufCLHpYdC6XeWGIa4TulhuO08+ZtQ8iSZv0uGcRhZkQ=";
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
