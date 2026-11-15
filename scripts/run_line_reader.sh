#!/usr/bin/env bash
# The struck text read line by line, the ceiling of the annotator's pixels (the plan of 2026-11-15, work/literature_revision_reader20261113/
# 整行读_上限方案与判定规则_2026-11-15.md).  Nothing is taught: the untaught Qwen3-VL-8B reads, loops stopped.
#  1. data.line_set merge: the annotator's line review (REVIEW, the export of the line page) laid over the dataset; pack: each line's
#     before (its readable regions' strike white, its unreadable regions white) and after (every region white), and its regions' crops.
#     Then nothing more until the images are looked at by hand (touch $R/lines_checked.done).
#  2. the regions read from their erased crops (erased_revision: R0), the lines from before and after apart (L1) and from the two in
#     one request (L2).
#  3. eval.line_reading: the three ways scored line by line and paired by writer; eval.transcription region by region, as a bridge
#     to the earlier table (the old readings of the regions the old pack held, and the new readings of the new set); the same scores in the
#     spelling before ÷ and · were brought to \div and \cdot, and how many labels and readings that spelling changes.
#  The rules, written before any reading (the plan's sixth part), on the lines all three read, AR and exact only: a line reading is
#  adopted when the upper end of its interval against R0 is not below 0 for both; of two, L1 unless L2 - L1 has an interval above 0;
#  the line helps only when the exact interval of the adopted one against R0 lies above 0.
# Unattended; a step whose marker is in $R is skipped.  CARDS (default "2": the user's word of 2026-11-15, card 2 only) are the cards,
# by index, a reader may go to; a reader has a card to itself and waits while another holds it.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; N=$O/paper2_clean; R=$O/paper2_line_reader20261115
DATA=$O/paper2_struck_dataset20261113; HM=${N}_revision_folds_human_masked; REVIEW=$O/paper2_line_review_returned20261115/export.json
MERGED=${N}_line_dataset20261115; LP=${N}_line_pack20261115; MODEL=/home/zsh/hme/models/Qwen3-VL-8B-Instruct
OLD=$O/paper2_layer_reader20261113/masked_untaught
mkdir -p "$R"
exec 8>"$R/lock"; flock -n 8 || { echo 'another run is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
PICK=(); for i in ${CARDS:-2}; do PICK+=("${ALL_CARDS[$i]}"); done
card() {   # a card of ours with room and no other job of ours (the 8B fills a card), waiting for one
    local c found
    until found=$(for c in "${PICK[@]}"; do [ "$(ours_on "$c")" -eq 0 ] && [ "$(nvidia-smi -i "$c" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 22000 ] && { echo "$c"; break; }; done; true) && [ -n "$found" ]; do sleep 60; done
    echo "$found"
}
read_pack() {   # OUT PACK CONDITIONS...: the untaught 8B on a card of its own
    local out="$1" pack="$2" c; shift 2
    [ -e "$out/complete.json" ] && return 0
    c=$(card); bash "$ROOT/scripts/launch_gpu_job.sh" "$c" "$(numa_of "$c")" "$out" -- "$PY" -m revrender.recognize_lines --pack "$pack" --model "$MODEL" \
        --backend hf --batch-size 8 --loop-limit 4 --conditions "$@" --output-dir "$out" > "$out.log" 2>&1
    test -e "$out/complete.json"
}

if ! done_ packs; then
    [ -e "$MERGED/run_manifest.json" ] || "$PY" -m revrender.data.line_set merge --dataset "$DATA/regions.json" --folds "$HM" --review "$REVIEW" --output-dir "$MERGED" > "$R/pack.log" 2>&1
    [ -e "$LP/run_manifest.json" ] || "$PY" -m revrender.data.line_set pack --dataset "$MERGED/regions.json" --lines "$MERGED/lines.json" --pages "$DATA/pages" \
        --pages "$O/paper2_full_review20261010/images" --output-dir "$LP" >> "$R/pack.log" 2>&1
    mark packs; fi
done_ lines_checked || { stamp 'the line pack is built: look at the images, then touch lines_checked.done and run again'; exit 0; }

if ! done_ read; then
    read_pack "$R/read_regions" "$LP/regions" erased_revision
    read_pack "$R/read_lines" "$LP" before_line after_line difference_line
    mark read; fi

if ! done_ score; then
    [ -e "$R/scores" ] || "$PY" -m revrender.eval.line_reading --pack "$LP" --regions "$R/read_regions" --lines "$R/read_lines" --output-dir "$R/scores" > "$R/score.log" 2>&1
    [ -e "$R/bridge_old.json" ] || "$PY" -m revrender.eval.transcription --pack "$HM" --arm "old=$OLD@erased_revision" --reference old --output "$R/bridge_old.json" > "$R/bridge_old.log" 2>&1
    [ -e "$R/bridge_new.json" ] || "$PY" -m revrender.eval.transcription --pack "$LP/regions" --arm "new=$R/read_regions@erased_revision" --reference new --output "$R/bridge_new.json" > "$R/bridge_new.log" 2>&1
    [ -e "$R/old_spelling" ] || "$PY" - "$R" "$LP" "$HM" "$OLD" <<'PYTHON' > "$R/old_spelling.log" 2>&1
import json, sys
from pathlib import Path
from revrender.data import ocr_lines
from revrender.eval import line_reading, transcription
from revrender.eval.coverage import prepare
R, LP, HM, OLD = map(Path, sys.argv[1:])
texts = {'labels': [r['text'] for r in json.loads((LP/'regions'/'pack.json').read_text())['rows'] if r['readable']] + [r['text'] for r in json.loads((LP/'pack.json').read_text())['rows']],
         'readings': [json.loads(x)['text'] for d in (R/'read_regions', R/'read_lines') for x in (d/'predictions.jsonl').read_text().splitlines()]}
new = {k: [prepare(t) for t in v] for k, v in texts.items()}
ocr_lines._SAME = tuple(p for p in ocr_lines._SAME if p[0] not in '÷·⋅')            # the spelling before 2026-11-15
changed = {k: sum(prepare(t) != n for t, n in zip(v, new[k])) for k, v in texts.items()}
(R/'old_spelling').mkdir()
(R/'old_spelling'/'changed.json').write_text(json.dumps({k: {'changed': changed[k], 'of': len(v)} for k, v in texts.items()}))
print('changed by the spelling', changed)
line_reading.main(['--pack', str(LP), '--regions', str(R/'read_regions'), '--lines', str(R/'read_lines'), '--output-dir', str(R/'old_spelling'/'scores')])
transcription.main(['--pack', str(HM), '--arm', f'old={OLD}@erased_revision', '--reference', 'old', '--output', str(R/'old_spelling'/'bridge_old.json')])
transcription.main(['--pack', str(LP/'regions'), '--arm', f"new={R/'read_regions'}@erased_revision", '--reference', 'new', '--output', str(R/'old_spelling'/'bridge_new.json')])
PYTHON
    mark score; fi
touch "$R/COMPLETE"; stamp 'line reading complete'
