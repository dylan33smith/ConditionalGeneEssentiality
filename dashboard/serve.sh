#!/usr/bin/env bash
# Serve the project dashboard on localhost (view it via an SSH tunnel).
# Usage:  ./dashboard/serve.sh [PORT]      (default 8080)
#
# Then, from your LAPTOP:
#   ssh -L 8080:localhost:8080 ds85@trojai2.luddy.indiana.edu
#   open http://localhost:8080
set -euo pipefail
PORT="${1:-8080}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Serving $DIR on http://127.0.0.1:$PORT  (Ctrl-C to stop)"
echo "Tunnel from your laptop:  ssh -L $PORT:localhost:$PORT ds85@trojai2.luddy.indiana.edu"
cd "$DIR"
exec python -m http.server "$PORT" --bind 127.0.0.1
