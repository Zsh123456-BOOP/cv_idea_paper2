#!/usr/bin/env bash
# The dedicated-OCR comparison (plan of 2026-11-21, 专用OCR对照; the user: 「可以，开工，4点都听你的」).  Inference only, nothing taught.
#   O-原图 / O-擦线: Unlimited-OCR-2 (the hme env, its page prompt), MiniCPM-OCR-1.0 and GLM-OCR (the new env; "Text Recognition:" and
#                    "Formula Recognition:" each) read every region's scan crop and the annotator's erased crop (revrender.recognize_ocr);
#   Q-照读 / Q-照读-擦线 / Q-指令 / Q-指令+行: Qwen3-VL-8B without an adapter under raw_neutral, erased_neutral, raw_revision and
#                    raw_context_revision (recognize_lines); the two that cannot refuse have their empty readings written as refusals
#                    (eval.ocr_compare normalize);
#   R0, 当前最好 (the sixth round's V-修) and H: the readings there are.
# eval.transcription against R0; eval.ocr_compare analyse (P1-P4, the ladder, the halves).  PART=check: the plan's section-five checks
# on printed non-project images (a pack of them: Chinese, formulas, both on one line) and the safeguards' self-test in both envs;
# PART=main: the 610 regions.  Cards 1 and 2 (the user: 「那就用两张卡，1，2」), one job at a time on each; 0 and 3 not touched.
# The hme env is used as it is (nothing installed or changed).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/home/zsh/hme/outputs; N=$O/paper2_clean; B=${N}_ocr_compare20261121; PART=${PART:-main}
case "$PART" in main) R=$B ;; check) R=${B}_check ;; *) echo "PART is main or check"; exit 2 ;; esac
mkdir -p "$R/checks"
P2=${N}_nafnet_folds20261119; V=${N}_revision_folds_human_masked20261117b_views; RAWV=${P2}_raw_views; FIX=${N}_covered_view20261121
NEW=/home/zsh/hme/envs/paper2_newmodels_20261121/bin/python; HME=$PY; MODELS=/home/zsh/hme/models; QWEN=$MODELS/Qwen3-VL-8B-Instruct
exec 8>"$R/lock"; flock -n 8 || { echo 'another run of this part is still going'; exit 1; }
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
trap 'stamp "failed at line $LINENO"; touch "$R/FAILED"' ERR
CARDS=("${ALL_CARDS[1]}" "${ALL_CARDS[2]}")
place() { place_on_cards "$@"; }
job() { local out="$1"; shift 2; bash "$ROOT/scripts/launch_gpu_job.sh" "$CARD" "$NUMA" "$out" -- "$@" > "$out.log" 2>&1; }
ocr() {   # OUT PYTHON KIND MODEL PROMPT IMAGE PACK
    [ -e "$1/predictions.jsonl" ] && [ -e "$1/run_manifest.json" ] || job "$1" -- "$2" -m revrender.recognize_ocr --pack "$7" --model-dir "$MODELS/$4" --kind "$3" \
        --prompt "$5" --image "$6" --output-dir "$1"
}
qwen() {  # OUT PACK CONDITION
    [ -e "$1/complete.json" ] || job "$1" -- "$HME" -m revrender.recognize_lines --pack "$2" --model "$QWEN" --backend hf --batch-size 8 --conditions "$3" \
        --loop-limit 4 --output-dir "$1"
}
declare -A LINE=([unlimited]="unlimited_ocr|Unlimited-OCR-2||$HME" [minicpm_text]="minicpm_ocr|MiniCPM-OCR-1.0|Text Recognition:|$NEW"
                 [minicpm_formula]="minicpm_ocr|MiniCPM-OCR-1.0|Formula Recognition:|$NEW" [glm_text]="glm_ocr|GLM-OCR|Text Recognition:|$NEW"
                 [glm_formula]="glm_ocr|GLM-OCR|Formula Recognition:|$NEW")
declare -A NAME=([unlimited]=Unlimited-OCR-2 [minicpm_text]="MiniCPM-OCR-1.0 Text" [minicpm_formula]="MiniCPM-OCR-1.0 Formula" [glm_text]="GLM-OCR Text" [glm_formula]="GLM-OCR Formula")
read_line() {  # LINE IMAGE PACK OUT
    IFS='|' read -r kind model prompt python <<< "${LINE[$1]}"
    place 12000 0 ocr "$4" "$python" "$kind" "$model" "$prompt" "$2" "$3"
}

if [ "$PART" = check ]; then
    if ! done_ checks; then
        # a pack of printed, non-project lines: Chinese, formulas, and both on one line (each part drawn on its own and joined, so the
        # Chinese is not lost to the formula font)
        [ -e "$R/check_pack/pack.json" ] || "$HME" - "$R/check_pack" <<'PY'
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
out = Path(sys.argv[1]); (out/'crops').mkdir(parents=True)
def draw(text, math):
    fig = plt.figure(figsize=(0.01, 0.01), dpi=100); fig.text(0, 0, text, fontsize=28, family=['WenQuanYi Zen Hei', 'DejaVu Sans'], parse_math=math)
    fig.canvas.draw(); path = out/'part.png'; fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1, facecolor='white'); plt.close(fig)
    return np.asarray(Image.open(path).convert('L'))
def join(parts):
    h = max(p.shape[0] for p in parts); padded = [np.pad(p, (((h-p.shape[0])//2, h-p.shape[0]-(h-p.shape[0])//2), (0, 0)), constant_values=255) for p in parts]
    return Image.fromarray(np.concatenate(padded, 1))
lines = {'cn': [('解得 x = 2，所以面积为 12', False)], 'formula': [(r'$x^{2}+3x-10=0$', True)], 'short': [(r'$-2$', True)],
         'mixed': [('由 ', False), (r'$\frac{a}{b}=\frac{3}{4}$', True), (' 得 ', False), (r'$a=3$', True)]}
rows = []
for name, parts in lines.items():
    image = join([draw(t, m) for t, m in parts]); path = f'crops/{name}.png'; image.save(out/path)
    rows.append({'id': f'check_{name}', 'line_id': f'check_{name}', 'writer': 'none', 'fold': 0, 'readable': True, 'mode': 'erase',
                 'text': ''.join(t for t, _ in parts), 'images': {'raw': path, 'erased': path}})
(out/'part.png').unlink(); (out/'pack.json').write_text(json.dumps({'schema': 'revrender-line-pack', 'rows': rows}, ensure_ascii=False, indent=1))
print(json.dumps([r['text'] for r in rows], ensure_ascii=False))
PY
        # the safeguards' self-test in both envs: eval from a file under transformers_modules refused, the standard library's works,
        # connections refused
        for env in "$HME" "$NEW"; do
            "$env" - "$R/checks" "$(basename "$(dirname "$(dirname "$env")")")" <<'PY'
import collections, importlib.util, json, socket, sys
from pathlib import Path
from revrender.recognize_ocr import safeguards
out, env = Path(sys.argv[1]), sys.argv[2]; safeguards()
test = out/f'guard_{env}'/'transformers_modules'; test.mkdir(parents=True); (test/'calls_eval.py').write_text('def run():\n    return eval("1+1")\n')
spec = importlib.util.spec_from_file_location('calls_eval', test/'calls_eval.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
try: mod.run(); blocked = False
except RuntimeError as e: blocked = 'SAFEGUARD' in str(e)
stdlib = collections.namedtuple('T', 'a')(1).a == 1
try: socket.create_connection(('example.com', 80)); net = False
except RuntimeError as e: net = 'SAFEGUARD' in str(e)
r = {'env': env, 'model_code_eval_refused': blocked, 'stdlib_eval_works': stdlib, 'network_refused': net}
(out/f'check6_safeguards_{env}.json').write_text(json.dumps(r, indent=2)); print(json.dumps(r)); sys.exit(0 if blocked and stdlib and net else 'safeguards not as they should be')
PY
        done
        for line in unlimited minicpm_text minicpm_formula glm_text glm_formula; do read_line "$line" raw "$R/check_pack" "$R/check_read_$line"; done; wait
        for line in unlimited minicpm_text minicpm_formula glm_text glm_formula; do test -s "$R/check_read_$line/predictions.jsonl"; done
        "$HME" - "$R" <<'PY'
import hashlib, json, sys
from pathlib import Path
R = Path(sys.argv[1]); pack = json.loads((R/'check_pack'/'pack.json').read_text())['rows']; sha = {r['id']: hashlib.sha256((R/'check_pack'/r['images']['raw']).read_bytes()).hexdigest() for r in pack}
out = {}
for d in sorted(R.glob('check_read_*')):
    rows = [json.loads(l) for l in (d/'predictions.jsonl').read_text().splitlines()]
    out[d.name] = {'readings': {r['id']: {'raw': r['raw'], 'text': r['text'], 'finish': r['finish_reason']} for r in rows},
                   'images_unchanged': all(r['image_sha256'] == sha[r['id']] for r in rows), 'all_read': len(rows) == len(pack)}
(R/'checks'/'check1_2_printed.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)); print(json.dumps(out, ensure_ascii=False)[:3000])
PY
        mark checks; fi
    touch "$R/COMPLETE"; stamp "check complete"; exit 0
fi

# The readings (one job at a time on each card).
if ! done_ reads; then
    for line in unlimited minicpm_text minicpm_formula glm_text glm_formula; do
        read_line "$line" raw "$V" "$R/read_raw_$line"; read_line "$line" erased "$V" "$R/read_erased_$line"
    done
    place 22000 1 qwen "$R/read_q_neutral" "$V" raw_neutral; place 22000 1 qwen "$R/read_q_neutral_erased" "$V" erased_neutral
    place 22000 1 qwen "$R/read_q_revision" "$V" raw_revision; place 22000 1 qwen "$R/read_q_context" "$RAWV" raw_context_revision; wait
    for d in "$R"/read_*; do test -e "$d/predictions.jsonl"; done
    # checks 2 and 3: the images given are the pack's files; every reading there, why each stopped
    "$HME" - "$R" "$V" <<'PY'
import collections, hashlib, json, sys
from pathlib import Path
R, V = Path(sys.argv[1]), Path(sys.argv[2]); rows = json.loads((V/'pack.json').read_text())['rows']
sha = {(r['id'], k): hashlib.sha256((V/r['images'][k]).read_bytes()).hexdigest() for r in rows for k in ('raw', 'erased')}
out = {}
for d in sorted(R.glob('read_*')):
    got = [json.loads(l) for l in (d/'predictions.jsonl').read_text().splitlines()]
    entry = {'readings': len(got), 'finish': dict(collections.Counter(g.get('finish_reason') for g in got))}
    if 'image_sha256' in got[0]:
        image = 'erased' if 'erased' in d.name else 'raw'; entry['images_are_the_pack_files'] = all(g['image_sha256'] == sha[(g['id'], image)] for g in got)
    out[d.name] = entry
ok = all(v['readings'] == len(rows) and v.get('images_are_the_pack_files', True) for v in out.values())
(R/'checks'/'check2_3_readings.json').write_text(json.dumps({'regions': len(rows), 'arms': out, 'passed': ok}, indent=2)); print(json.dumps(out))
sys.exit(0 if ok else 'a reading set is short or an image is not the pack file')
PY
    mark reads; fi

if ! done_ score; then
    for q in q_neutral q_neutral_erased; do [ -e "$R/norm_$q/predictions.jsonl" ] || "$HME" -m revrender.eval.ocr_compare normalize --readings "$R/read_$q" --output-dir "$R/norm_$q"; done
    dirs5() { local k s=; for k in 0 1 2 3 4; do s+="${s:+,}$1_f$k"; done; echo "$s"; }
    ARMS_=()
    for line in unlimited minicpm_text minicpm_formula glm_text glm_formula; do
        ARMS_+=(--arm "O-原图|${NAME[$line]}=$R/read_raw_$line" --arm "O-擦线|${NAME[$line]}=$R/read_erased_$line")
    done
    ARMS_+=(--arm "Q-照读=$R/norm_q_neutral" --arm "Q-照读-擦线=$R/norm_q_neutral_erased" --arm "Q-指令=$R/read_q_revision" --arm "Q-指令+行=$R/read_q_context"
            --arm "R0=$(dirs5 "$P2/read_R0")" --arm "当前最好=$(dirs5 "$FIX/read_fix")" --arm "H=$(dirs5 "$P2/read_H")")
    [ -e "$R/scores_vs_R0.json" ] || "$HME" -m revrender.eval.transcription --pack "$V" "${ARMS_[@]}" --reference R0 --output "$R/scores_vs_R0.json" > "$R/scores_vs_R0.log" 2>&1
    [ -e "$R/analysis/analysis.json" ] || "$HME" -m revrender.eval.ocr_compare analyse --pack "$V" "${ARMS_[@]}" --shares "$FIX/checks/strike_shares.json" \
        --output-dir "$R/analysis" > "$R/analysis.log" 2>&1
    mark score; fi
touch "$R/COMPLETE"; stamp "main complete"
