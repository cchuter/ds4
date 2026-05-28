#!/usr/bin/env bash
# tests/test_gpu_args_cli.sh — smoke test that the four binaries wire
# up --gpu-vram / --gpu-devices correctly. Run from the repo root via
# `make test`. Does NOT exercise any CUDA hardware.
set -uo pipefail

cd "$(dirname "$0")/.."

PASS=0
FAIL=0
LOG=$(mktemp)

ok()   { PASS=$((PASS+1)); echo "ok $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

assert_grep() {
    # $1 = name, $2 = pattern, $3 = file
    if grep -q -- "$2" "$3" 2>/dev/null; then ok "$1"; else
        fail "$1 (pattern not in $3)"
        echo "    --- content of $3 ---"
        head -20 "$3" | sed 's/^/    /'
    fi
}

# Binaries to check
BINS=(./ds4 ./ds4-server ./ds4-bench ./ds4-agent)
NAMES=(ds4 ds4-server ds4-bench ds4-agent)

# 1: each binary's --help mentions both flags.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    if [ ! -x "$bin" ]; then
        fail "$name not built — skipping help check"
        continue
    fi
    "$bin" --help > "$LOG" 2>&1 || true
    assert_grep "$name --help mentions --gpu-vram" "gpu-vram" "$LOG"
    assert_grep "$name --help mentions --gpu-devices" "gpu-devices" "$LOG"
done

# 2: parser error on syntactically invalid value. For ds4-bench, we
# also pass --prompt-file /dev/null so it doesn't exit on the
# "specify exactly one of --prompt-file or --chat-prompt-file" check
# before the gpu-vram parser is reached.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    if [ "$name" = "ds4-bench" ]; then
        "$bin" --gpu-vram abc -m /dev/null --prompt-file /dev/null > "$LOG" 2>&1
    else
        "$bin" --gpu-vram abc -m /dev/null > "$LOG" 2>&1
    fi
    rc=$?
    if [ $rc -eq 0 ]; then
        fail "$name --gpu-vram abc should exit non-zero (got 0)"
    else
        ok "$name --gpu-vram abc exits non-zero ($rc)"
    fi
    # Confirm the error message mentions --gpu-vram (parser was reached).
    if grep -q "gpu-vram" "$LOG" 2>/dev/null; then
        ok "$name --gpu-vram abc error mentions the flag"
    else
        # Some binaries use a logger that doesn't echo flag names —
        # accept any stderr indicating a parse problem.
        if grep -qE "not a number|abc" "$LOG"; then
            ok "$name --gpu-vram abc error message reasonable"
        else
            fail "$name --gpu-vram abc error message is unclear"
            head -10 "$LOG" | sed 's/^/    /'
        fi
    fi
done

# 3: count mismatch.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    "$bin" --gpu-vram 40,12 --gpu-devices 0 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        ok "$name count-mismatch errors out ($rc)"
    else
        fail "$name count-mismatch should error"
    fi
done

# 4: --cuda --help still works (the flag alone shouldn't break parsing).
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    "$bin" --cuda --help > "$LOG" 2>&1 || true
    # Servers may print a usage banner; check help still surfaced.
    if grep -qE "Usage:|usage:|--help" "$LOG"; then
        ok "$name --cuda --help still prints help"
    else
        fail "$name --cuda --help did not print help text"
    fi
done

# 5: --gpu-vram 0 short-circuit. We use ds4 (CLI) specifically because
# it produces predictable stdout/stderr.
if [ -x ./ds4 ]; then
    ./ds4 --gpu-vram 0 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        ok "ds4 --gpu-vram 0 exits non-zero (expected: model-load fail)"
    else
        fail "ds4 --gpu-vram 0 returned 0 — unexpected"
    fi
    # The layout line must NOT appear (short-circuit happens before).
    if grep -q "GPU config:" "$LOG"; then
        fail "ds4 --gpu-vram 0 should NOT print GPU layout line"
        head -10 "$LOG" | sed 's/^/    /'
    else
        ok "ds4 --gpu-vram 0 does not print GPU layout (short-circuit reached)"
    fi
fi

# 6: --gpu-vram 40,12 layout line.
if [ -x ./ds4 ]; then
    ./ds4 --gpu-vram 40,12 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if grep -q "GPU config: 2 devices \[0,1\] requested, budgets 40,12 GB" "$LOG"; then
        ok "ds4 --gpu-vram 40,12 prints expected layout line"
    else
        fail "ds4 --gpu-vram 40,12 missing or malformed layout line"
        head -10 "$LOG" | sed 's/^/    /'
    fi
fi

rm -f "$LOG"

echo ""
echo "test_gpu_args_cli: PASS=$PASS FAIL=$FAIL"
if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
