#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
docker build -t padlab:ubuntu22 .
docker build --target artifacts --output type=local,dest=artifacts/ubuntu22 .
echo "Qt5 Ubuntu 22.04 binaries: $project_dir/artifacts/ubuntu22/bin"
