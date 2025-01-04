{
  lib,
  # build deps
  buildPythonPackage,
  fetchFromGitHub,
  # test deps
  pytest,
  cython,
  # runtime deps
  deprecation,
  contourpy,
  pywavelets,
  numpy,
  scipy,
  scikit-image,
  python3Packages,
}:
let
  helper = import ./version_patch.nix { inherit lib python3Packages; };
in
buildPythonPackage {
  pname = "centrosome";
  version = "1.3.0";

  src = fetchFromGitHub {
    owner = "CellProfiler";
    repo = "centrosome";
    rev = "570b3f8c9568b3ffe1743f3a9fe3b223cc43fcb4";
    sha256 = "sha256-w0rrMFovPzsg02J8oMZKaSYk7l/kqsNeSY44q1dQ8KI=";
  };

  # Relax deps constraints
  postPatch = ''
    substituteInPlace setup.py \
      ${(helper.patchPackageVersions ../src/subpackages/library/pyproject.toml).subStr}
  '';

  buildInputs = [
    pytest
    cython
  ];

  propagatedBuildInputs = [
    deprecation
    contourpy
    pywavelets
    numpy
    scipy
    scikit-image
  ];
  pythonImportsCheck = [ ];

  meta = {
    description = "Centrosome";
    homepage = "https://cellprofiler.org";
    license = lib.licenses.bsd3;
  };
}
