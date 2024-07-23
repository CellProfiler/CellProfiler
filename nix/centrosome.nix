{ pkgs }:
pkgs.python3Packages.buildPythonPackage {
  pname = "centrosome";
  version = "1.2.23";

  src = pkgs.fetchFromGitHub {
    owner = "CellProfiler";
    repo = "centrosome";
    rev = "5cdaa5a";
    sha256 = "sha256-cXzji/KQAQ6uY43IDis8WGl1s+Czz/S1BEhuRnYBJjY=";
  };

  # Relax deps constraints
  postPatch = ''
    substituteInPlace setup.py \
      --replace "scipy>=1.4.1,<1.11" "scipy>=1.4.1" \
      --replace "matplotlib>=3.1.3,<3.8" "matplotlib>=3.1.3"
  '';

  buildInputs = with pkgs.python3Packages; [
    pytest
    cython
  ];

  propagatedBuildInputs = with pkgs.python3Packages; [
    deprecation
    contourpy
    pywavelets
    numpy
    scipy
    scikit-image
  ];
  pythonImportsCheck = [];

  meta = with pkgs.lib; {
    description = "Centrosome";
    homepage = "https://cellprofiler.org";
    # license = license.mit;
  };

}
