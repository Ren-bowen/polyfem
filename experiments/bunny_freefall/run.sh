#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/home/bowen/polyfem-diffipc-original/build-make/PolyFEM_bin -j run.json --max_threads 8 > run.log 2>&1
/home/bowen/miniconda3/envs/env_gipc/bin/python export.py
