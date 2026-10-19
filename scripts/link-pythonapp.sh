#!/bin/bash

PREFIX="${CONDA_PREFIX}"

if [ ! -d "${CONDA_PREFIX}/python.app" ]; then
    (
    export PREFIX
    
    # run the post-link script which creates the python.app directory
    ${PREFIX}/bin/.python.app-post-link.sh
    )
fi
