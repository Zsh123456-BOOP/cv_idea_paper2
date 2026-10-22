#!/usr/bin/env bash
# The restorer experiments read again on a wider validation (user 2026-10-22): the 16 fixed regions and 41 strikes plus the
# development half of the test questions (configs/test_halves.json; the annotator's export of the test page) -- every version
# run_annotation_value.sh and run_learning_curve.sh weighed, evaluated together per stage, chosen on validation and development
# pooled, and the stages compared as before.  Weights unchanged; nothing trained.  The sealed half is never read.
#   run_wider_validation.sh EXPORT
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_wider_validation20261022
VAL=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
export DEV_EXPORT="$1" DEV_PAGES=$O/paper2_test_text20261018_page/site/images CHOOSE_SPLITS="val dev"
test -f "$DEV_EXPORT"
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"
declare -A LISTS=([synthetic]=$O/paper2_annotation_value20261022/candidates_synthetic.json [r201]=$O/paper2_annotation_value20261022/candidates_finetuned.json
                  [r50]=$O/paper2_learning_curve20261022/candidates_all_50.json [r100]=$O/paper2_learning_curve20261022/candidates_all_100.json)
for stage in synthetic r50 r100 r201; do
    done_ "$stage" || { evaluate_versions "${LISTS[$stage]}" "${N}_wide_versions_$stage" "$R/$stage.log"; mark "$stage"; }
done
for stage in r50 r100 r201; do
    done_ "compare_$stage" || { "$PY" -m revrender.eval.unified_restoration compare --before "${N}_wide_versions_synthetic/chosen" \
        --after "${N}_wide_versions_$stage/chosen" --output-dir "$R/compare_$stage" > "$R/compare_$stage.log" 2>&1; mark "compare_$stage"; }
done
touch "$R/COMPLETE"; stamp complete
