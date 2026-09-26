#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /home/bowen/miniconda3/envs/env_gipc/bin/python visualize.py "$@"
