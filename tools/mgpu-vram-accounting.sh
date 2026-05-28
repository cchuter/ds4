#!/usr/bin/env bash
# mgpu-vram-accounting.sh — reads a bench harness log file (and its
# matching .smi) and prints a per-device VRAM-accounting markdown
# table comparing engine's printed `(used / budget)` to nvidia-smi.
#
# Usage:
#   tools/mgpu-vram-accounting.sh path/to/split-label-timestamp.log
#
# Reads the matching .smi (same prefix, .smi extension).
# Per the v0 design doc, expected delta is ≤ 256 MB per device.
set -uo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <bench-log>" >&2
    exit 2
fi

log="$1"
smi="${log%.log}.smi"

if [ ! -f "$log" ]; then
    echo "log not found: $log" >&2
    exit 1
fi

echo "## VRAM accounting — $(basename "$log")"
echo

# Pull engine-printed layout block.
echo "### engine-reported layout"
awk '/multi-GPU layout:/,/^ds4: peer access matrix|^ds4: CPU-spill/' "$log" \
    | grep -E "GPU[0-9]+|CPU" \
    | sed 's/^/    /'

echo
echo "### nvidia-smi snapshot"
if [ -f "$smi" ]; then
    echo
    printf "| %-5s | %-20s | %-20s |\n" "idx" "used (MiB)" "free (MiB)"
    printf "| %-5s | %-20s | %-20s |\n" "---" "---" "---"
    while IFS=',' read -r idx used free; do
        idx=$(echo "$idx" | xargs)
        used=$(echo "$used" | xargs)
        free=$(echo "$free" | xargs)
        printf "| %-5s | %-20s | %-20s |\n" "$idx" "$used" "$free"
    done < "$smi"
else
    echo "    (no .smi snapshot found at $smi)"
fi
