#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${PADLAB_IMAGE:-padlab:ubuntu22}"
user_id="$(id -u)"
group_id="$(id -g)"
args=(run --rm --network none --user "$user_id:$group_id"
  --security-opt label=disable
  --mount "type=bind,source=$project_dir,target=/data"
  --workdir /data)
if [[ -n "${WAYLAND_DISPLAY:-}" && -S "${XDG_RUNTIME_DIR:-}/$WAYLAND_DISPLAY" ]]; then
  args+=(--tmpfs "/run/padlab:rw,mode=700,uid=$user_id,gid=$group_id"
    --mount "type=bind,source=$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY,target=/run/padlab/$WAYLAND_DISPLAY,readonly"
    -e XDG_RUNTIME_DIR=/run/padlab -e "WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
    -e QT_QPA_PLATFORM=wayland)
elif [[ -n "${DISPLAY:-}" ]]; then
  args+=(--mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly
    -e "DISPLAY=$DISPLAY" -e QT_QPA_PLATFORM=xcb -e QT_X11_NO_MITSHM=1)
  auth_file="${XAUTHORITY:-$HOME/.Xauthority}"
  if [[ -f "$auth_file" ]]; then
    args+=(--mount "type=bind,source=$auth_file,target=/tmp/padlab.xauth,readonly" -e XAUTHORITY=/tmp/padlab.xauth)
  fi
else
  echo "No Wayland/X11 display in this session. Run from a desktop terminal." >&2
  exit 1
fi
exec docker "${args[@]}" "$image" padlab "$@"
