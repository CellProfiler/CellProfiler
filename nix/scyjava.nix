{
  lib,
  # build deps
  buildPythonPackage,
  fetchPypi,
  # test deps
  pytest,
  # runtime deps
  jgo,
  jpype1,
}:
buildPythonPackage rec {
  pname = "scyjava";
  version = "1.10.0";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-f4boY36Oed5W1kPZeOgyU/84l0pD12V583ktXgdtVoE=";
  };

  pyproject = true;

  buidInputs = [
    pytest
  ];

  propagatedBuildInputs = [
    jpype1
    jgo
  ];

  pythonImportsCheck = ["scyjava"];

  meta = {
    description = "⚡ Supercharged Java access from Python ⚡";
    homepage = "https://github.com/scijava/scyjava";
    license = lib.licenses.unlicense;
  };
}
