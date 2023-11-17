# CellProfiler Docker containers

This folder contains the CellProfiler docker containers. `dockerhub` contains information and code for pushing images dockerhub. The remaining are containers intended for use in development of CellProfiler. 

The CellProfiler development containers allow for CellProfiler dependencies to be isolated in a container while using the local CellProfiler repo. This allows you to make changes to CellProfiler during development, but avoiding potentially tricky dependency installation on your system. 

There are two development containers included here:
1. `docker/dev`
   1. This is intended to be used with the VSCode extension [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), but can also be used to run CellProfiler in **headless** mode.
2. `docker/dev_gui`
   1. Launches a desktop environment at `http://localhost:9876/` that allows the CellProfiler GUI to be used in the browser. 

For **building and running** the above containers, there the same assumption for both: that the [CellProfiler repo](https://github.com/CellProfiler/CellProfiler) is cloned and that your terminal is in the root of this repo, as follows:

```
CellProfiler <- **Local terminal directory should be here**
└── docker/
    ├── dev/
    │   ├── Dockerfile
    │   └── prepare_env.sh
    └── dev_gui/
        └── Dockerfile
```

## Building containers locally

Build the headless docker container with the following command:
`docker buildx build --platform linux/amd64 -t cellprofiler_dev:latest CellProfiler/docker/dev`

Build the GUI enabled docker container with the following command:
`docker buildx build --platform linux/amd64 -t cellprofiler_dev_gui:latest docker/dev_gui`

## Run the headless container and set up

Run the headless container as follows:
1. `docker run -it --rm -v $(PWD):/workspace --name cellprofiler_dev cellprofiler_dev:latest`
2. Once the docker bash terminal opens, run the following command to install your local CellProfiler repo: `bash docker/dev/prepare_env.sh`
3. Attach to the running docker container with the [VSCode Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/attach-container#:~:text=To%20attach%20to%20a%20Docker,you%20want%20to%20connect%20to.)
4. If you would like to use Jupyter notebooks, you will need to install the [VSCode Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) in VSCode while inside your dev container to do this. 
5. VSCode may not recognise that Python is installed. If this is the case, press `ctrl+shift+p` in VSCode and type `Python: Select Interpreter` and enter the Python path `/opt/conda/bin/python`. Jupyter notebooks should now work from inside the dev container in VSCode.

## Run the CellProfiler GUI container

1. `docker run -it --rm -v $(PWD):/workspace --name cellprofiler_dev_gui -p 9876:9876 cellprofiler_dev_gui:latest`
2. In your browser, go to [http://localhost:9876/](http://localhost:9876/)
3. In the terminal window enter the command `bash docker/dev/prepare_env.sh`
4. Start CellProfiler gui by entering the command `cellprofiler` in the terminal window
