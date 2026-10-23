#!/usr/bin/env bash
# Should a question go in whole?  (user 2026-10-23)  The 19 test questions whose every region was confirmed (the 2026-10-22 export):
# their students split, seed 42, about a third held out.  NAFNet fine-tuned as the prefill was ([finetune_long] from the same chosen
# synthetic version) on the 201 training regions plus the other questions -- as region crops, or whole (every tile, the ink outside
# the regions kept) -- and read on the held-out questions whole or only inside the detector's regions (eval.page_restoration), beside
# the prefill model itself.  Unattended; a step whose marker is in $R is skipped.  Cards 1 and 2.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_whole_question20261023
E=$O/paper2_test_export20261022/export.json; P=$O/paper2_test_text20261018_page/site/images
V4=${N}_restoration_pairs20261007_v4; SET=${N}_v4_finetune_set; SEL1=${N}_v4_versions_synthetic; TOOL=${N}_finetune_long_NAFNet
VAL=$O/paper2_fixed_validation20261007; UP=$O/paper2_restoration_sources20261008; STAIN_UP=$O/paper2_stainrestorer_resources20261009/upstream
C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; C3=$CARD3
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this experiment is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
source "$ROOT/scripts/_restorers.sh"

if ! done_ split; then
    "$PY" - "$E" "$R" <<'PY'
import json, random, sys
from revrender.util import SEED
export, out = sys.argv[1:]
pages = json.load(open(export))['pages']
full = sorted(k for k, p in pages.items() if p.get('revisions') and all(r.get('confirmed') and (r.get('text_confirmed') or r.get('unreadable')
              or r.get('not_revision')) for r in p['revisions']))
writers = sorted({k.split('_')[0] for k in full}); random.Random(SEED).shuffle(writers); held = set(writers[:round(len(writers)/3)])
open(out+'/train_ids.txt', 'x').write('\n'.join(k for k in full if k.split('_')[0] not in held)+'\n')
open(out+'/held_ids.txt', 'x').write('\n'.join(k for k in full if k.split('_')[0] in held)+'\n')
print(len(full), 'questions;', len(held), 'of', len(writers), 'students held out')
PY
    mark split; fi
if ! done_ data; then
    for arm in crops whole; do
        flag=(); [ "$arm" = crops ] || flag=(--whole)
        "$PY" -m revrender.data.restoration_pairs questions --export "$E" --pages "$P" --ids "$R/train_ids.txt" --review-set "$SET" --source "$V4" \
            "${flag[@]}" --output-dir "${N}_whole_q_data_$arm" >> "$R/data.log" 2>&1
    done
    mark data; fi
if ! done_ train; then
    INIT=$(chosen_of "$SEL1/chosen" NAFNet)
    ( finetune_restorer NAFNet "$C1" 0 "$INIT" "${N}_whole_q_data_crops" "${N}_whole_q_crops_NAFNet" finetune_long ) > "$R/train_crops.log" 2>&1 & A=$!
    ( finetune_restorer NAFNet "$C2" 1 "$INIT" "${N}_whole_q_data_whole" "${N}_whole_q_whole_NAFNet" finetune_long ) > "$R/train_whole.log" 2>&1 & B=$!
    wait $A; wait $B; mark train; fi
if ! done_ evaluate; then
    step() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]+"/selection.json"))["checkpoint"])' "$1"; }
    models=(); for spec in "prefill:$TOOL:step_364" "crops:${N}_whole_q_crops_NAFNet:$(step "${N}_whole_q_crops_NAFNet")" \
                          "whole:${N}_whole_q_whole_NAFNet:$(step "${N}_whole_q_whole_NAFNet")"; do
        IFS=: read -r name run st <<< "$spec"; models+=(--model "${name}_boxes=$run:$st:boxes" --model "${name}_page=$run:$st:page"); done
    bash "$ROOT/scripts/launch_gpu_job.sh" "$C1" 0 "$R/scores" -- "$PY" -m revrender.eval.page_restoration --export "$E" --pages "$P" \
        --ids "$R/held_ids.txt" --upstream "$UP" "${models[@]}" --output-dir "$R/scores" > "$R/evaluate.log" 2>&1
    mark evaluate; fi
touch "$R/COMPLETE"; stamp complete
