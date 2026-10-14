#!/usr/bin/env bash
# Build a model in set mode and render it with the real LDraw parts.
#
#   build_and_render.sh <mesh.glb> <out_dir> [facade.npz] [extra semantic args...]
#
# Writes <out_dir>/model.ldr, report.json, build.log and render.png (five
# views: lon 0, 45, 90, 180, 270), then prints the structure numbers.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

mesh=$1; out=$2; shift 2
facade_args=()
if [[ $# -gt 0 && $1 == *.npz ]]; then facade_args=(--facade "$1"); shift; fi

mkdir -p "$out"
.venv/bin/everything2lego semantic "$mesh" -o "$out" --max-studs 56 --n-samples 50000 \
    --set-mode "${facade_args[@]}" "$@" > "$out/build.log" 2>&1 || true
[[ -f $out/model.ldr ]] || { tail -20 "$out/build.log"; exit 1; }

.venv/bin/python tools/render_build.py "$out/model.ldr" "$out/render.png" --size 600

.venv/bin/python - "$out/report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
def find(d, keys, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if any(s in k for s in keys) and not isinstance(v, (dict, list)):
                print(f"  {p}: {v}")
            find(v, keys, p)
find(r, ("parts", "components", "floating", "unsupported", "added_supports",
         "set:walls", "set:gables", "set:piers", "inliers"))
EOF
