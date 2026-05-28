#!/usr/bin/env bash
# bench-mgpu-v0.sh — run the v0 multi-GPU bench matrix and capture
# per-split logs + nvidia-smi readings. Designed to be run on the box
# after `make cuda`. Does NOT modify ds4-bench; just invokes it with
# the new --gpu-vram flag from mgpu-cli-wiring.
#
# Each split runs independently. A single-split failure (e.g. CPU-spill
# refusal when the model doesn't fit) is recorded in the log and the
# next split is attempted. The harness itself exits 0 if it reached the
# end without an internal scripting error.
#
# Env overrides:
#   DS4_BENCH_MODEL      path to GGUF model (default: IQ2XXS on box)
#   DS4_BENCH_PROMPT     path to bench prompt file
#   DS4_BENCH_REPORT_DIR where per-run logs land
#   DS4_LOCK_FILE        ds4 instance lock (do not collide with user's
#                         running server on the same host)
#   DS4_BENCH_CTX_MAX    --ctx-max for ds4-bench (default 8192)
#   DS4_BENCH_GEN_TOKENS --gen-tokens (default 64)
set -uo pipefail

MODEL=${DS4_BENCH_MODEL:-/home/cchuter/work/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
PROMPT_FILE=${DS4_BENCH_PROMPT:-tests/long_context_story_prompt.txt}
REPORT_DIR=${DS4_BENCH_REPORT_DIR:-.dev-team/reports/mgpu-bench}
LOCK=${DS4_LOCK_FILE:-/tmp/ds4-bench-mgpu.lock}
CTX_MAX=${DS4_BENCH_CTX_MAX:-8192}
GEN_TOKENS=${DS4_BENCH_GEN_TOKENS:-64}

mkdir -p "$REPORT_DIR"

# Sanity checks before running.
if [ ! -x ./ds4-bench ]; then
    echo "bench-mgpu-v0: ./ds4-bench is not built; run \`make cuda\` first" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "bench-mgpu-v0: model not found: $MODEL" >&2
    exit 1
fi
if [ ! -f "$PROMPT_FILE" ]; then
    echo "bench-mgpu-v0: prompt file not found: $PROMPT_FILE" >&2
    echo "bench-mgpu-v0: set DS4_BENCH_PROMPT to override" >&2
    exit 1
fi

results=()

run_split () {
    local label="$1"; shift
    local flags="$*"
    local ts
    ts=$(date -u +%Y%m%d-%H%M%S)
    local log="$REPORT_DIR/${label}-${ts}.log"
    local smi="$REPORT_DIR/${label}-${ts}.smi"

    echo "=== Running $label ($flags) ==="
    {
        echo "# $label"
        echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# flags: $flags"
        echo "# model: $MODEL"
        echo "# ctx-max: $CTX_MAX gen-tokens: $GEN_TOKENS"
        echo
    } > "$log"

    # Launch in background so we can poll for "ready" and capture
    # nvidia-smi at the right moment.
    DS4_LOCK_FILE="$LOCK" ./ds4-bench \
        --prompt-file "$PROMPT_FILE" \
        -m "$MODEL" \
        --cuda $flags \
        --ctx-start 2048 \
        --ctx-max "$CTX_MAX" \
        --gen-tokens "$GEN_TOKENS" \
        >> "$log" 2>&1 &
    local pid=$!

    # Wait up to 180 s for "backend initialized for graph diagnostics"
    # which signals engine load complete. Then snapshot nvidia-smi.
    local captured_smi=0
    for _i in $(seq 1 180); do
        if grep -q "backend initialized for graph diagnostics" "$log" 2>/dev/null; then
            sleep 2
            nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader > "$smi" 2>/dev/null || true
            captured_smi=1
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    wait "$pid"
    local rc=$?

    if [ "$captured_smi" -eq 0 ]; then
        # If we never saw the readiness line, still capture nvidia-smi
        # post-mortem so the report has SOMETHING to compare against.
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader > "$smi" 2>/dev/null || true
    fi

    # Extract the peer-matrix line if present.
    local peer_line
    peer_line=$(grep "peer access matrix" "$log" 2>/dev/null | head -1 || true)

    # Extract the "multi-GPU layout:" block if present.
    local layout
    layout=$(awk '/multi-GPU layout:/,/^ds4: peer access matrix/' "$log" 2>/dev/null | grep -E "GPU[0-9]+|CPU" || true)

    # Detect outcome:
    #   - "CPU-spill placement detected" → CPU-spill refusal
    #   - "ctx_tokens,prefill_tokens" header present → bench ran
    #   - otherwise → error or aborted
    local outcome
    local prefill_tps="" gen_tps="" ctx_tokens=""
    if grep -q "CPU-spill placement detected" "$log"; then
        outcome="ENV-BLOCK (CPU-spill refusal — wave 3b required)"
    elif grep -q "^ctx_tokens,prefill_tokens" "$log"; then
        outcome="OK (bench ran)"
        # Extract the bench CSV row: ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
        local csv_row
        csv_row=$(grep -E '^[0-9]+,[0-9]+,[0-9.]+,[0-9]+,[0-9.]+,[0-9]+' "$log" | head -1)
        if [ -n "$csv_row" ]; then
            ctx_tokens=$(echo "$csv_row" | awk -F',' '{print $1}')
            prefill_tps=$(echo "$csv_row" | awk -F',' '{print $3}')
            gen_tps=$(echo "$csv_row" | awk -F',' '{print $5}')
        fi
    elif grep -q "another ds4 process is already running" "$log"; then
        outcome="LOCK CONFLICT (set DS4_LOCK_FILE to a free path)"
    else
        outcome="UNKNOWN (rc=$rc, see log)"
    fi

    results+=("$label|$outcome|$prefill_tps|$gen_tps|$ctx_tokens|$log|$smi")
    {
        echo
        echo "# === post-run summary ==="
        echo "# outcome: $outcome"
        echo "# peer-matrix: $peer_line"
        echo "# layout:"
        echo "$layout" | sed 's/^/#   /'
    } >> "$log"

    echo "  outcome: $outcome (log: $log)"
}

# The matrix from the spec.
run_split single-gpu     --gpu-vram 48
run_split split-24-24    --gpu-vram 24,24
run_split split-40-12    --gpu-vram 40,12
# CPU-only baseline: cap ctx_max harder so this completes in finite time.
DS4_BENCH_CTX_MAX_SAVED=$CTX_MAX
CTX_MAX=2048
run_split cpu-only       --gpu-vram 0
CTX_MAX=$DS4_BENCH_CTX_MAX_SAVED
run_split auto           --gpu-vram auto

# Final markdown summary to stdout.
echo
echo "## bench-mgpu-v0 summary"
echo
printf "| %-12s | %-12s | %-10s | %-9s | %s |\n" "split" "ctx_tokens" "prefill tps" "gen tps" "outcome"
printf "| %-12s | %-12s | %-10s | %-9s | %s |\n" "---" "---" "---" "---" "---"
for row in "${results[@]}"; do
    IFS='|' read -r label outcome prefill_tps gen_tps ctx_tokens _log _smi <<< "$row"
    printf "| %-12s | %-12s | %-10s | %-9s | %s |\n" \
        "$label" \
        "${ctx_tokens:--}" \
        "${prefill_tps:--}" \
        "${gen_tps:--}" \
        "$outcome"
done

echo
echo "Per-run logs and nvidia-smi snapshots in: $REPORT_DIR"
