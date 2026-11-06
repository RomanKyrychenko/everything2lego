#!/bin/bash
# Run a command, print its output, and append its wall time and peak memory to the analysis ledger.
#   tools/timed.sh <step-name> <command...>
# -> /Volumes/PRO-G40/lego_data/analysis/ledger.jsonl: {"step", "when", "seconds", "peak_mb", "cmd", "tail"} (tail: last 12 lines of output)
set -o pipefail
LEDGER=${LEDGER:-/Volumes/PRO-G40/lego_data/analysis/ledger.jsonl}
step=$1; shift
out=$(mktemp); tm=$(mktemp)
/usr/bin/time -l -o "$tm" "$@" 2>&1 | tee "$out"
rc=${PIPESTATUS[0]}
secs=$(awk '/real/ {print $1; exit}' "$tm")
peak=$(awk '/maximum resident set size/ {printf "%.0f", $1 / 1048576; exit}' "$tm")
/usr/bin/python3 - "$step" "$secs" "$peak" "$*" "$out" "$LEDGER" "$rc" <<'EOF'
import json, sys, time
step, secs, peak, cmd, out, ledger, rc = sys.argv[1:]
tail = [l for l in open(out, errors="replace").read().splitlines() if l.strip()][-12:]
rec = {"step": step, "when": time.strftime("%Y-%m-%d %H:%M"), "seconds": float(secs or 0), "peak_mb": int(peak or 0),
       "rc": int(rc), "cmd": cmd, "tail": tail}
open(ledger, "a").write(json.dumps(rec) + "\n")
print(f"[ledger] {step}: {rec['seconds']:.0f} s, peak {rec['peak_mb']} MB, rc {rc}")
EOF
rm -f "$out" "$tm"
exit $rc
