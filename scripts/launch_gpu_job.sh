#!/usr/bin/env bash
# One short GPU job (a readout or re-inference) on one card, holding the card's lock shared like the training launchers.
#   launch_gpu_job.sh GPU_UUID NUMA OUT_DIR_THAT_MUST_NOT_EXIST_YET -- command ...      (GPU_UUID: two cards joined by a comma for a job that spans both)
# The command runs with the card pinned, strict-determinism environment and the NUMA node bound; a failure touches $OUT.FAILED.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
GPU_UUID="$1"; NUMA="$2"; OUT="$3"; shift 3
test "$1" = -- || { echo 'usage: GPU NUMA OUT -- command'; exit 2; }; shift
refuse_card3 "$GPU_UUID"
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
trap 'echo "job failed at line $LINENO $(date -u +%FT%TZ)"; touch "$OUT.FAILED"' ERR
hold_card "$GPU_UUID"
export HME_NUMA_NODE="$NUMA"
echo "GPU lock held (shared) $(date -u +%FT%TZ): $*"
numactl --cpunodebind="$NUMA" --membind="$NUMA" "$@"
test -e "$OUT"
echo "job complete $(date -u +%FT%TZ)"
