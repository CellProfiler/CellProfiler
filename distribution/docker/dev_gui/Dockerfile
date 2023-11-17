# docker buildx build --platform linux/amd64 -t ctromanscoia/cellprofiler_dev_gui:latest docker/dev_gui
# docker run -it --rm -v $(PWD):/workspace --name cellprofiler_dev_gui -p 9876:9876 cellprofiler/cellprofiler_dev_gui:latest

#### Dockerfile from docker/dev start
FROM mambaorg/micromamba:1.5.1-jammy

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN micromamba install -y -n base -c conda-forge \ 
    python=3.8.* \
    freetype \
    gcc \
    gxx \
    git \
    mysql \
    && micromamba clean -qya

RUN micromamba install -y -n base -c conda-forge -c bioconda \
    boto3~=1.28.41 \
    centrosome==1.2.2 \
    docutils==0.15.2 \
    h5py~=3.6.0 \
    imageio~=2.31.3 \
    inflect~=7.0.0 \
    Jinja2~=3.1.2 \
    joblib~=1.3.2 \
    mahotas~=1.4.13 \
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
    ipykernel \
    && micromamba clean -qya

# Matplotlib is unable to be resolved by micromamba at current version for some reason
RUN pip install matplotlib~=3.1.3

# 4.2.1 leads to "DistributionNotFound". However, 4.2.0 has GUI issues
# pkg_resources.DistributionNotFound The 'wxPython==4.2.0' distribution was not found and is required by CellProfiler
# RUN pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04 wxPython==4.2.1
RUN pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04 wxPython==4.2.0

ENV HOME=/workspace
WORKDIR $HOME

#### Dockerfile from docker/dev end
#### Could replace the above lines by using `FROM cellprofiler/cellprofiler_dev`

# Inherited container sets the user as mambauser, but we need root for build
USER root

# Install Xpra and dependencies
RUN apt-get update \
    && apt-get install -y \
    wget \
    gnupg2 \
    apt-transport-https \
    libgtk-3-dev \
    libnotify-dev \
    libsdl2-dev \
    libwebkit2gtk-4.0-dev \
    && wget -O - https://xpra.org/gpg.asc | apt-key add - \
    && echo "deb https://xpra.org/ jammy main" > /etc/apt/sources.list.d/xpra.list

RUN apt-get update && \
    apt-get install -yqq \
        xpra \
        xvfb \
        xterm \
        sshfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:100
ENV XPRA_PORT=9876
## Alternative start if cellprofiler pip installed during build
# ENV XPRA_START="python -m cellprofiler"
## Start terminal command that can be used to source install CellProfiler
ENV XPRA_START="xterm"
ENV XPRA_EXIT_WITH_CLIENT="yes"
ENV XPRA_XVFB_SCREEN="1920x1080x24+32"
EXPOSE 9876

## Attempt at making start menu/desktop cellprofiler shortcuts
# RUN mkdir -p ~/.local/share/applications && \
#     echo "[Desktop Entry]" >> ~/.local/share/applications/cellprofiler.desktop && \
#     echo "Name=CellProfiler" >> ~/.local/share/applications/cellprofiler.desktop && \
#     # echo "Exec=python -m cellprofiler" >> ~/.local/share/applications/cellprofiler.desktop && \
#     echo "Exec=xterm" >> ~/.local/share/applications/cellprofiler.desktop && \
#     echo "Icon=cellprofiler" >> ~/.local/share/applications/cellprofiler.desktop && \
#     echo "Type=Application" >> ~/.local/share/applications/cellprofiler.desktop && \
#     echo "Categories=Application;Development;" >> ~/.local/share/applications/cellprofiler.desktop && \
#     echo "Terminal=false" >> ~/.local/share/applications/cellprofiler.desktop

# ADD ./CellProfiler-favicon-120.png /usr/share/applications/ellProfiler-favicon-120.png
# ADD ./cellprofiler.desktop /usr/share/applications/cellprofiler.desktop

CMD echo "######## Connect here: http://localhost:$XPRA_PORT ########"; \
    xpra start \
    --bind-tcp=0.0.0.0:$XPRA_PORT \
    --html=on \
    --start="$XPRA_START" \
    --exit-with-client="$XPRA_EXIT_WITH_CLIENT" \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 $XPRA_XVFB_SCREEN -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --file-transfer=on \
    --bell=no \
    $DISPLAY

# Back to non-root priv
USER $MAMBA_USER
