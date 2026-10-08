#!/usr/bin/env bash
# Wait for all three lanes; evaluate frozen checkpoints once; render offline comparison.
# Usage: SUITE UPSTREAM
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
SUITE="$1"; UPSTREAM="$2"
require_clean_source
trap 'touch "$SUITE/EVALUATION_FAILED"' ERR
for lane in 0 2 3; do
  while [ ! -f "$SUITE/lane$lane/COMPLETED" ]; do
    if [ -f "$SUITE/lane$lane/FAILED" ]; then
      echo "training lane $lane failed; test evaluation will not run"
      touch "$SUITE/EVALUATION_FAILED"
      exit 3
    fi
    if [ -f "$SUITE/lane$lane.pid" ] && ! kill -0 "$(cat "$SUITE/lane$lane.pid")" 2>/dev/null; then
      echo "training lane $lane exited without COMPLETED"
      touch "$SUITE/EVALUATION_FAILED"
      exit 3
    fi
    if [ "$SECONDS" -gt 172800 ]; then
      echo '48-hour evaluation wait exceeded; training processes left untouched'
      touch "$SUITE/EVALUATION_FAILED"
      exit 3
    fi
    sleep 60
  done
done
bash "$ROOT/scripts/eval_restoration_pilot.sh" --suite "$SUITE" "$UPSTREAM" "$SUITE/evaluation"
PYTHONPATH="$ROOT" "$PY" -m revrender.tools.restoration_report \
  --evaluation "$SUITE/evaluation" --expected-models 6 --output-dir "$SUITE/report"
touch "$SUITE/COMPLETED"
