#!/usr/bin/env bash
# Wait for all three lanes; evaluate frozen checkpoints once; render offline comparison.
# Usage: SUITE UPSTREAM [RECOVER_ON_PHYSICAL_CARD]
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
SUITE="$1"; UPSTREAM="$2"
require_clean_source
trap 'code=$?; if [ "$code" -ne 0 ]; then touch "$SUITE/EVALUATION_FAILED"; fi' EXIT
CARD="${3:-3}"
if [ "$#" -ge 3 ]; then
  case "$CARD" in 0|2|3) ;; *) exit 2 ;; esac
  # Explicit continuation of unfinished lanes; completed model directories are immutable.
  if [ -f "$SUITE/EVALUATION_FAILED" ]; then
    mv "$SUITE/EVALUATION_FAILED" "$SUITE/EVALUATION_FAILED.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  for lane in 0 2 3; do
    [ ! -f "$SUITE/lane$lane/COMPLETED" ] || continue
    if [ -f "$SUITE/lane$lane.pid" ] && kill -0 "$(cat "$SUITE/lane$lane.pid")" 2>/dev/null; then
      echo "lane $lane is still running; refusing duplicate launch"
      exit 3
    fi
    bash "$ROOT/scripts/run_restoration_suite.sh" "$lane" \
      /home/zsh/hme/outputs/paper2_restoration_pairs20261007_v4 "$UPSTREAM" "$SUITE/lane$lane" resume "$CARD"
  done
fi
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
bash "$ROOT/scripts/eval_restoration_pilot.sh" --suite "$SUITE" "$UPSTREAM" "$SUITE/evaluation" "$CARD"
PYTHONPATH="$ROOT" "$PY" -m revrender.tools.restoration_report \
  --evaluation "$SUITE/evaluation" --expected-models 6 --output-dir "$SUITE/report"
touch "$SUITE/COMPLETED"
