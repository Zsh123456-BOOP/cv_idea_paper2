#!/usr/bin/env bash
# Re-run the three analysis readings on the eight frozen validation models -- the mask grids' ceilings and the thickness of
# the delivered masks (eval.representation), the six-class pixel errors (eval.pixel_errors) and the scoring on ink alone
# (--ink-only of eval.fates, eval.ap and eval.bootstrap) -- and compare each with paper 1's record.  Run it from a deployed snapshot.
#
#   scripts/verify_analysis.sh OUT_ROOT REF_DIR
#
#   OUT_ROOT  a directory that must not exist yet
#   REF_DIR   paper 1's records: freeze_<model>.json for rvr maskrcnn pointrend condinst solov2 mask2former maskdino dimaskdino,
#             and representation.json, classes.json, ink_only.json
#
# CPU only; about 15 minutes.  No test split is read.
set -uo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"; REF="$2"
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir -p "$OUT"
B="${BUNDLE:-/home/zsh/hme/work/revision_bundle878_gold_v2/morphology_s42}"
BS="${BUNDLE_SHA:-574da74b6444699a4c375077c0aa9ff5819569692ba134a32db50d17da8acddf}"
MODELS="rvr maskrcnn pointrend condinst solov2 mask2former maskdino dimaskdino"
cd "$ROOT"
export PYTHONPATH="$ROOT"
cpu() { CUDA_VISIBLE_DEVICES= "$PY" "$@"; }
FREEZES=$(for m in $MODELS; do echo --freeze "$m=$REF/freeze_$m.json"; done)
VARIANTS=$(for m in $MODELS; do echo --variant "$m=freeze:$REF/freeze_$m.json"; done)
PAIRS=$(for m in $MODELS; do [ "$m" = rvr ] || echo --pair "rvr-$m"; done)
run() {   # name, then the module and its arguments
    local name="$1"; shift
    cpu -m "$@" --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
    echo "$name exit=$?" >> "$OUT/summary.log"
}

run representation revrender.eval.representation --bundle "$B" --bundle-sha "$BS" $FREEZES --workers 6 &
run pixel_errors revrender.eval.pixel_errors --bundle "$B" --bundle-sha "$BS" $FREEZES --workers 6 &
( run fates_ink revrender.eval.fates --bundle "$B" --bundle-sha "$BS" $FREEZES --ink-only
  run ap_ink revrender.eval.ap --bundle "$B" --bundle-sha "$BS" $FREEZES --ink-only
  run bootstrap_ink revrender.eval.bootstrap --bundle "$B" --bundle-sha "$BS" $VARIANTS $PAIRS --resamples 5000 --workers 6 --ink-only ) &
wait

cpu - "$OUT" "$REF" >> "$OUT/summary.log" <<'EOF'
import json, sys
from pathlib import Path
out, ref = Path(sys.argv[1]), Path(sys.argv[2])
load = lambda p: json.loads(Path(p).read_text())
checks = {}

old, new = load(ref/'representation.json'), load(out/'representation.json')
checks['representation_equal_paper1'] = all(new[k] == old[k] for k in (
    'gold_revisions', 'width_edges', 'grids', 'grid_definitions', 'per_revision_iou', 'widths', 'thickness_steps_px', 'thickness'))

old, new = load(ref/'classes.json'), load(out/'pixel_errors.json')
checks['pixel_classes_equal_paper1'] = all(new[k] == old[k] for k in ('classes', 'ink_below', 'near_px')) \
    and set(new['models']) <= set(old['models']) and all(new['models'][m] == old['models'][m] for m in new['models'])

old = load(ref/'ink_only.json')['models']
fates, ap, pairs = load(out/'fates_ink.json')['models'], load(out/'ap_ink.json')['models'], load(out/'bootstrap_ink.json')['pairs']
def ink_only_equal(m):
    o, f = old[m], fates[m]
    region, ink, new = o['region'], o['ink'], f['ink']
    gold = new['gold_components']
    counts = {k: new['gold'].get(k, 0) for k in ('matched', 'extent', 'missed')}
    same = (region['true_positive'], region['false_positive'], region['gold'], region['f1']) \
        == (f['true_positive'], f['false_positive'], f['gold_components'], f['f1'])
    same &= (ink['true_positive'], ink['false_positive'], ink['gold'], ink['f1']) == (new['true_positive'], new['false_positive'], gold, new['f1'])
    same &= (ink['AP'], ink['AP50'], ink['AP75']) == tuple(ap[m]['ink'][k] for k in ('AP', 'AP50', 'AP75'))
    same &= (ink['found'], ink['delineated']) == (new['found'], new['drawn_right'])
    same &= ink['found_not_delineated'] == counts['extent']/gold and ink['missed'] == counts['missed']/gold and ink['fates'] == counts
    same &= (ink['gold_without_ink'], ink['delivered_without_ink']) == (new['without_ink']['gold'], new['without_ink']['delivered'])
    if 'ours_minus_this_f1' in ink:
        paired = pairs[f'rvr-{m}']['f1']
        same &= ink['ours_minus_this_f1'] == {'difference': paired['difference'], 'ci95': paired['ci95']}
    return same
checks['ink_only_equal_paper1'] = set(old) == set(fates) and all(ink_only_equal(m) for m in old)
print('CHECKS ' + json.dumps(checks, sort_keys=True))
EOF
echo "---- summary ($OUT/summary.log)"; cat "$OUT/summary.log"
