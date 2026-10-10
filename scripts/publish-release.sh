#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
repo="${1:-gabapppp/padlab}"
version=0.1.0
tag="v$version"
release_dir="$project_dir/artifacts/ubuntu22/release"
if command -v gh >/dev/null 2>&1; then
  gh_cmd="$(command -v gh)"
elif [[ -x "$project_dir/.tools/gh" ]]; then
  gh_cmd="$project_dir/.tools/gh"
else
  echo "Install GitHub CLI first: https://cli.github.com/" >&2
  exit 1
fi
for asset in "padlab-$version-ubuntu22.04-amd64.deb" "padlab-$version-ubuntu22.04-amd64.tar.gz" SHA256SUMS; do
  [[ -f "$release_dir/$asset" ]] || { echo "Missing $asset. Run ./scripts/build-docker.sh first." >&2; exit 1; }
done
(cd "$release_dir" && sha256sum --check SHA256SUMS)
git diff --quiet && git diff --cached --quiet || { echo "Commit source changes first." >&2; exit 1; }
[[ "$(git rev-parse "$tag^{commit}")" == "$(git rev-parse HEAD)" ]] || { echo "HEAD must match $tag." >&2; exit 1; }
"$gh_cmd" auth status >/dev/null 2>&1 || "$gh_cmd" auth login --hostname github.com --git-protocol https --web --skip-ssh-key
"$gh_cmd" auth setup-git --hostname github.com
if ! "$gh_cmd" repo view "$repo" >/dev/null 2>&1; then
  "$gh_cmd" repo create "$repo" --public --description "Qt5 + OpenCV pipeline workbench with subpixel boundary measurements"
fi
remote_url="https://github.com/$repo.git"
if git remote get-url origin >/dev/null 2>&1; then
  existing="$(git remote get-url origin)"
  [[ "$existing" == "$remote_url" || "$existing" == "git@github.com:$repo.git" ]] || { echo "origin points to a different repository: $existing" >&2; exit 1; }
else
  git remote add origin "$remote_url"
fi
git push -u origin HEAD:main
git push origin "$tag"
"$gh_cmd" release create "$tag" --repo "$repo" --verify-tag \
  --title "PadLab $tag — Ubuntu 22.04 x64" --notes-file packaging/RELEASE_NOTES.md \
  "$release_dir/padlab-$version-ubuntu22.04-amd64.deb" \
  "$release_dir/padlab-$version-ubuntu22.04-amd64.tar.gz" \
  "$release_dir/SHA256SUMS"
