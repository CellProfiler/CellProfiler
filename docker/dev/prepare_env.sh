#!/usr/bin/env bash

# Install local core and cellprofiler repos
pip install -e core/
pip install -e cellprofiler

# Enter the micromamba environment
/bin/bash /usr/local/bin/_dockerfile_shell.sh bash