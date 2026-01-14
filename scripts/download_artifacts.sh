#!/usr/bin/env bash

mkdir "${PIXI_PROJECT_ROOT}"/artifacts
run_id=$(gh run list --workflow meiosis -L 1 --json databaseId -q ".[0].databaseId")
gh run download "$run_id" --dir "${PIXI_PROJECT_ROOT}"/artifacts
mv -vf "${PIXI_PROJECT_ROOT}"/artifacts/*/* "${PIXI_PROJECT_ROOT}"/dist/
rmdir "${PIXI_PROJECT_ROOT}"/artifacts/*
rmdir "${PIXI_PROJECT_ROOT}"/artifacts/
