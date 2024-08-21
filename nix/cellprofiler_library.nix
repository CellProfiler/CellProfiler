{
  lib,
  # build deps
  buildPythonPackage,
  fetchPypi,
  setuptools-scm,
  # test deps
  pytest,
  # runtime deps
  centrosome,
  matplotlib,
  numpy,
  mahotas,
  scikit-image,
  scipy,
  packaging,
  cp_version,
}:
buildPythonPackage rec {
  pname = "cellprofiler_library";
  version = cp_version;

  src = ../src/subpackages/library;
  pyproject = true;

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-warn "numpy~=1.24.4" "numpy<=1.26.4" \
      --replace-warn "scikit-image~=0.20.0" "scikit-image>=0.17.2,<=0.22.0" \
      --replace-warn "scipy>=1.9.1,<1.11" "scipy>=1.9.1,<=1.13"
    echo 'fallback_version = "${version}"' >> pyproject.toml
  '';

  buildInputs = [
    pytest
    (
      setuptools-scm.overrideAttrs rec {
        version = "8.1.0";
        src = fetchPypi {
          pname = "setuptools_scm";
          inherit version;
          hash = "sha256-Qt6htldxy6k7elFdZaZdgkblYHaKZrkQalksjn8myKc=";
        };
      }
    )
  ];

  propagatedBuildInputs = [
    numpy
    scikit-image
    scipy
    mahotas
    centrosome
    matplotlib
    packaging
  ];

  pythonImportsCheck = ["cellprofiler_library"];

  meta = {
    description = "Cellprofiler library";
    homepage = "https://github.com/CellProfiler/CellProfiler";
    license = lib.licenses.bsd3;
  };
}
