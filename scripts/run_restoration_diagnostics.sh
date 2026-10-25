#!/usr/bin/env bash
# The zero-training diagnostics of the plan "只减不加" (work/model_plan20261025/保字擦线_只减不加_方案_2026-10-25.md, section 2):
# eval.restoration_diagnostics pack on the deciding batch of the stroke version (its chosen versions' crops; D1 needs a card), then
# the pack read twice by the evaluation's OCR engine (the second reading measures the reading's own spread), then scored.  Judged as
# written before the readings: D1 holds if its paired difference with RevSep is above +1; D3 if at least two of its three are above +1.
# Unattended; a step whose marker is in $R is skipped.  Cards: whichever have room (_common.sh open_card).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_restoration_diagnostics20261025; D=${N}_restoration_diagnostics
E=${N}_revsep_stroke_versions_final; UP=$O/paper2_restoration_sources20261008
DEV_EXPORT=$O/paper2_test_export20261022/export.json; DEV_PAGES=$O/paper2_test_text20261018_page/site/images
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
index() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="$1" '$2==u{print $1}'; }

if ! done_ pack; then
    until card=$(open_card 8000); do sleep 30; done; [ "$card" != "$CARD3" ] || export ALLOW_GPU3=1
    bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$(numa_of "$card")" "$D" -- "$PY" -m revrender.eval.restoration_diagnostics pack --evaluation "$E" \
        --upstream "$UP" --dev-export "$DEV_EXPORT" --dev-pages "$DEV_PAGES" --output-dir "$D" > "$R/pack.log" 2>&1
    mark pack; fi
for reading in qwen8b qwen8b_repeat; do
    if ! done_ "$reading"; then
        until a=$(open_card 14000) && b=$(open_card 14000 "$a"); do sleep 30; done
        case "$a,$b" in *314fdf43*) export ALLOW_GPU3=1 ;; esac
        bash "$ROOT/scripts/launch_gpu_job.sh" "$a,$b" 0 "$D/$reading" -- env CUDA_VISIBLE_DEVICES="$(index "$a"),$(index "$b")" \
            "$PY" -m revrender.recognize_lines --pack "$D" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct \
            --conditions erased_neutral --tp 2 --gpu-memory .55 --output-dir "$D/$reading" > "$R/$reading.log" 2>&1
        mark "$reading"; fi
done
done_ score || { "$PY" -m revrender.eval.restoration_diagnostics score --diagnostics "$D" --readings "$D/qwen8b" "$D/qwen8b_repeat" \
    --output "$R/scores.json" > "$R/score.log" 2>&1; mark score; }
touch "$R/COMPLETE"; stamp complete
