#!/usr/bin/env bash
# Launches the poly-wan-video server. Extra arguments are passed to "wan-video serve".
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found at .venv. See README.md for setup."
    exit 1
fi

source ".venv/bin/activate"
exec wan-video serve --port 7860 "$@"
