{
  lib,
  # build deps
  buildPythonPackage,
  fetchPypi,
  # test deps
  pytest,
  # runtime deps
  psutil,
  setuptools,
}:
buildPythonPackage rec {
  pname = "jgo";
  version = "1.0.6";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-MYJNMp3FgklUzBUKnZNMVDoieBjcHA6m6iB0HC5GKyQ=";
  };

  pyproject = true;

  buidInputs = [
    pytest
  ];

  propagatedBuildInputs = [
    psutil
    setuptools
  ];

  pythonImportsCheck = ["jgo"];

  meta = {
    description = "Launch Java code from the CLI, installation-free. ☕";
    homepage = "https://github.com/scijava/jgo";
    license = lib.licenses.unlicense;
  };
}
