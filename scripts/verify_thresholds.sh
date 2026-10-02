#!/usr/bin/env bash
# Re-run the mask-threshold freeze of the native baselines from paper 1's frozen checkpoints and compare each result with
# paper 1's record: the 0.5 re-run must reproduce the frozen archive line by line (checked inside the run), the readings at
# 0.3 / 0.4 / 0.5 and the chosen point must equal the paper's.   Run it from a deployed snapshot.
#
#   scripts/verify_thresholds.sh OUT_ROOT REF_DIR [METHODS...]
#
#   OUT_ROOT  a directory that must not exist yet
#   REF_DIR   copies of paper 1's freeze_<model>.json (the table's records)
#   METHODS   default: maskrcnn pointrend condinst solov2 mask2former maskdino dimaskdino
#
# Three lanes, one per card 0 / 1 / 2 (shared lock, one job at a time); about 12 minutes per model, mostly on the CPU.
set -uo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"; REF="$2"; shift 2
METHODS=("$@"); if [ "${#METHODS[@]}" -eq 0 ]; then METHODS=(maskrcnn pointrend condinst solov2 mask2former maskdino dimaskdino); fi
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir -p "$OUT"
B="${BUNDLE:-/home/zsh/hme/work/revision_bundle878_gold_v2/morphology_s42}"
BS="${BUNDLE_SHA:-574da74b6444699a4c375077c0aa9ff5819569692ba134a32db50d17da8acddf}"
OLD_NATIVE="${PAPER1_NATIVE:-/home/zsh/hme/work/revision_native_goldv2_v1}"
CARDS=(GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0 GPU-d0bcc564-5bd9-b943-f995-8536415088e3 GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0)
NUMAS=(0 0 1)
cd "$ROOT"
export PYTHONPATH="$ROOT"

one() {   # method, lane
    local m="$1" lane="$2" up="$UPSTREAM_V1"
    case "$m" in solov2|dimaskdino) up="$UPSTREAM_V2" ;; esac
    bash scripts/launch_gpu_job.sh "${CARDS[$lane]}" "${NUMAS[$lane]}" "$OUT/$m" -- "$NATIVE_PY" -m revrender.baselines.native.threshold \
        --method "$m" --freeze "$OLD_NATIVE/$m/freeze.json" --bundle "$B" --bundle-sha "$BS" --upstream-root "$up" --weights-dir "$WEIGHTS" \
        --output-dir "$OUT/$m" > "$OUT/$m.log" 2>&1
    echo "$m exit=$?" >> "$OUT/summary.log"
}
lane() {  # lane index, then its methods
    local lane="$1"; shift
    for m in "$@"; do one "$m" "$lane"; done
}
buckets=("" "" "")
i=0; for m in "${METHODS[@]}"; do buckets[$((i%3))]+="$m "; i=$((i+1)); done
for lane in 0 1 2; do lane "$lane" ${buckets[$lane]} & done
wait

CUDA_VISIBLE_DEVICES= "$PY" - "$OUT" "$REF" "${METHODS[@]}" >> "$OUT/summary.log" <<'EOF'
import json, sys
from pathlib import Path
out, ref, methods = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
result = {}
for m in methods:
    try:
        new, paper = (json.loads(p.read_text()) for p in (out/m/'freeze.json', ref/f'freeze_{m}.json'))
    except FileNotFoundError:
        result[m] = False
        continue
    same = all(new['validation'][k] == paper['validation'][k] for k in ('own_best', 'by_mask_threshold', 'tp_within_fp_budget')) \
        and all(new[k] == paper[k] for k in ('mask_threshold', 'score_threshold', 'nms', 'checkpoint_sha256', 'chosen_step'))
    result[m] = same
print('THRESHOLDS ' + json.dumps(result, sort_keys=True))
EOF
echo "---- summary ($OUT/summary.log)"; cat "$OUT/summary.log"
