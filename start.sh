#!/bin/sh
# Installs the Python packages, runs first-time setup if needed, then starts Analytixa on http://127.0.0.1:4000
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
if [ ! -f config.env ]; then python3 setup.py || exit 1; fi
python3 server.py
