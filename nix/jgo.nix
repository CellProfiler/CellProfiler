{ pkgs }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "jgo";
  version = "1.0.6";

  src = pkgs.fetchPypi {
    inherit pname version;
    sha256 = "sha256-MYJNMp3FgklUzBUKnZNMVDoieBjcHA6m6iB0HC5GKyQ=";
  };

  pyproject = true;

  # Relax deps constraints
  # postPatch = ''
  #   substituteInPlace setup.py \
  #     --replace "scipy>=1.4.1,<1.11" "scipy>=1.4.1" \
  #     --replace "matplotlib>=3.1.3,<3.8" "matplotlib>=3.1.3"
  # '';


  buidInputs  = with pkgs.python3Packages; [
    pytest
  ];

  propagatedBuildInputs = with pkgs.python3Packages; [
    psutil
    setuptools
  ];

  pythonImportsCheck = [];

  meta = with pkgs.lib; {
    description = "Jgo";
    homepage = "https://github.com/scijava/scyjava";
    # license = license.mit;
  };

}
