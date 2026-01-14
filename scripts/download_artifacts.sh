#!/usr/bin/env bash

mkdir artifacts
run_id=$(gh run list --workflow meiosis -L 1 --json databaseId -q ".[0].databaseId")
gh run download "$run_id"
mv artifacts/*/* dist/
rmdir artifacts/*
rmdir artifacts/
