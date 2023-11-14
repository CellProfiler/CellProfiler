#!/usr/bin/env bash

# Install local CellProfiler repos
# We request no deps since we have defined them in the 
# Dockerfile. This allows us to pin dependencies required
# for the docker container base image (eg. wxPython 4.2.1, 
# not 4.2.0) without regard to compatibility with other 
# systems
pip install --no-cache-dir --no-deps -e src/subpackages/library
pip install --no-cache-dir --no-deps -e src/subpackages/core
pip install --no-cache-dir --no-deps -e src/frontend
