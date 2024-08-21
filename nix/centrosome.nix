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
}:
buildPythonPackage {
  pname = "centrosome";
  version = "1.2.3";

  src = fetchFromGitHub {
    owner = "CellProfiler";
    repo = "centrosome";
    rev = "5cdaa5a";
    sha256 = "sha256-cXzji/KQAQ6uY43IDis8WGl1s+Czz/S1BEhuRnYBJjY=";
  };

  # Relax deps constraints
  postPatch = ''
    substituteInPlace setup.py \
      --replace "scipy>=1.4.1,<1.11" "scipy>=1.4.1" \
      --replace "matplotlib>=3.1.3,<3.8" "matplotlib>=3.1.3" \
      --replace "PyWavelets<1.5" "PyWavelets<=1.6" \
      --replace "scikit-image>=0.17.2,<0.22.0" "scikit-image>=0.17.2,<=0.22.0" \
      --replace "contourpy<1.2.0" "contourpy<=1.2.0"
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
  pythonImportsCheck = [];

  meta = {
    description = "Centrosome";
    homepage = "https://cellprofiler.org";
    license = lib.licenses.bsd3;
  };
}
