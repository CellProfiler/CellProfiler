{ pkgs, cp_version, centrosome }:
pkgs.python3Packages.buildPythonPackage rec {
  pname = "cellprofiler_library";
  version = cp_version;

  src = ../src/subpackages/library;
  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace "centrosome~=1.2.2" "centrosome~=1.2.3"
    echo 'fallback_version = "${version}"' >> pyproject.toml
  '';

  buildInputs  = with pkgs.python3Packages; [
    pytest
    (
      setuptools-scm.overrideAttrs rec {
        version = "8.1.0";
        src = pkgs.fetchPypi {
          pname = "setuptools_scm";
          inherit version;
          hash = "sha256-Qt6htldxy6k7elFdZaZdgkblYHaKZrkQalksjn8myKc=";
        };

      }
    )
  ];

  propagatedBuildInputs = with pkgs.python3Packages; [
    numpy
    scikit-image
    scipy
    mahotas
    centrosome
    matplotlib
    packaging
  ];
  pythonImportsCheck = [];

  meta = with pkgs.lib; {
    description = "Cellprofiler library";
    homepage = "https://cellprofiler.org";
    # license = license.mit;
  };

}
