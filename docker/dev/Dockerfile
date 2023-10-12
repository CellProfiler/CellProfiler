# docker buildx build --platform linux/amd64 -t ctromanscoia/cellprofiler_dev:latest CellProfiler/docker/dev
# docker run -it --rm -v $(PWD):/workspace --name cellprofiler_dev ctromanscoia/cellprofiler_dev:latest sh CellProfiler/docker/dev/prepare_env.sh
# docker push ctromanscoia \cellprofiler_dev

FROM mambaorg/micromamba:1.5.1-bullseye-slim

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN micromamba install -y -n base -c conda-forge \
    python=3.9.* \
    freetype \
    gcc \
    gxx \
    git \
    mysql \
    && micromamba clean -qya


RUN pip install --no-cache-dir --upgrade pip \
    boto3~=1.28.41 \
    centrosome==1.2.2 \
    docutils==0.15.2 \
    h5py~=3.6.0 \
    imageio~=2.31.3 \
    inflect~=7.0.0 \
    Jinja2~=3.1.2 \
    joblib~=1.3.2 \
    mahotas~=1.4.13 \
    matplotlib~=3.1.3 \
    mysqlclient~=2.0.0 \
    numpy~=1.24.4 \
    Pillow~=10.0.0 \
    pyzmq~=22.3.0 \
    "sentry-sdk>=0.18.0,<=1.31.0" \
    requests~=2.31.0 \
    scikit-image~=0.20.0 \
    scikit-learn~=1.3.0 \
    "scipy>=1.9.1,<1.11" \
    scyjava>=1.9.1 \
    six~=1.16.0 \
    "tifffile>=2022.4.8,<2022.4.22" \
    rapidfuzz~=3.0.0 \
    future>=0.18.2 \
    fsspec>=2021.11.0 \
    lxml>=4.6.4 \
    psutil>=5.9.5 \
    pyzmq~=22.3.0 \
    zarr~=2.16.1 \
    google-cloud-storage~=2.10.0 \
    attrdict \
    pooch \
    black \
    pytest~=7.4.1 \
    Sphinx>=3.1.1 \
    sphinx-rtd-theme>=0.5.0 \
    pre-commit \ 
    pyinstaller \
    twine \
    ipykernel

RUN pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/debian-11/wxPython-4.2.0-cp39-cp39-linux_x86_64.whl

ENV HOME=/workspace
WORKDIR $HOME