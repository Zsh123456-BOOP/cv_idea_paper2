#!/usr/bin/env bash
# Re-run the port's equivalence checks on the server: the paper-1 model's inference and numbers, its training trace, the
# baselines' frozen predictions and entry points.  Run it from a deployed snapshot (scripts/deploy.sh).
#
#   scripts/verify_equivalence.sh OUT_ROOT REF_DIR [probe]
#
#   OUT_ROOT  a directory that must not exist yet; every check writes under it (/home/zsh/hme/outputs/paper2_*)
#   REF_DIR   copies of paper 1's records: ap883.json main_table.json bootstrap887.json fsr_analysis.json and
#             freeze_<model>.json for rvr maskrcnn pointrend condinst solov2 mask2former maskdino dimaskdino
#   probe     also train the formal run until its first validation (step 2000) and compare it with paper 1's curve (~20 min)
#
# Lanes run in parallel on cards 0, 1, 2 (one job each, shared lock) and the CPU; card 3 is not used.  The data, weights and
# paper-1 run locations are those of the server (override with the environment variables below).  No test split is read.
set -uo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"; REF="$2"; PROBE="${3:-}"
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir -p "$OUT"
B="${BUNDLE:-/home/zsh/hme/work/revision_bundle878_gold_v2/morphology_s42}"
BS="${BUNDLE_SHA:-574da74b6444699a4c375077c0aa9ff5819569692ba134a32db50d17da8acddf}"
C="${CORRECTED:-/home/zsh/hme/work/revision_corrected861_v1}"
CS="${CORRECTED_SHA:-65a2db20b0a4ea563c153b7c1a1b63c42f45e68c65b63df86212201edcd9e429}"
OLD="${PAPER1_RUN:-/home/zsh/hme/work/revision_tune880_v1/control}"          # the frozen model's run
OLD_NATIVE="${PAPER1_NATIVE:-/home/zsh/hme/work/revision_native_goldv2_v1}"   # the baselines' runs
CKPT_SHA="${CHECKPOINT_SHA:-9f8ae601f1db17e1bb2e164d2e63344756bc3719a44a0e47f7e720a43a978153}"
G0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; G1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; G2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0
MODELS="rvr maskrcnn pointrend condinst solov2 mask2former maskdino dimaskdino"
cd "$ROOT"
export PYTHONPATH="$ROOT"
cpu() { CUDA_VISIBLE_DEVICES= "$PY" "$@"; }
FREEZES=$(for m in $MODELS; do echo --freeze "$m=$REF/freeze_$m.json"; done)

lane_inference() {   # card 0: new inference on the frozen weights, read at the frozen point, compared line by line
    bash scripts/launch_gpu_job.sh $G0 0 "$OUT/infer" -- "$PY" -m revrender.infer --checkpoint "$OLD/formal/best.pt" \
        --checkpoint-sha "$CKPT_SHA" --bundle "$B" --bundle-sha "$BS" --output-dir "$OUT/infer" > "$OUT/infer.log" 2>&1
    echo "infer exit=$?" >> "$OUT/summary.log"
    cpu -m revrender.eval.report --bundle "$B" --bundle-sha "$BS" --freeze rvr="$OLD/freeze.json" \
        --archive rvr="$OUT/infer/dev_predictions.npz" --compare rvr="$OLD/formal/dev_predictions.npz" \
        --reference-ap "$REF/ap883.json" --reference-table "$REF/main_table.json" > "$OUT/infer_report.log" 2>&1
    echo "infer_report exit=$?" >> "$OUT/summary.log"
    cpu -m revrender.eval.freeze --bundle "$B" --bundle-sha "$BS" --run "$OLD/formal" --output "$OUT/freeze_rvr_rederived.json" \
        > "$OUT/freeze_rederive.log" 2>&1
    echo "freeze_rederive exit=$?" >> "$OUT/summary.log"
}

lane_training() {    # card 1: smoke, exact replay, learning gate; the smoke trace against paper 1's; optionally 2000 steps
    STOP_AFTER=learning bash scripts/launch_train_arm.sh control $G1 0 "$B" "$BS" "$C" "$CS" "$OUT/train" 8000 > "$OUT/train.launcher.log" 2>&1
    echo "train_gates exit=$?" >> "$OUT/summary.log"
    cpu -m revrender.tools.compare_trace "$OLD/smoke_a/trace.json" "$OUT/train/smoke_a/trace.json" > "$OUT/trace_compare.log" 2>&1
    echo "trace_compare exit=$?" >> "$OUT/summary.log"
    cpu -m revrender.tools.compare_trace "$OUT/train/smoke_a/trace.json" "$OUT/train/smoke_b/trace.json" > "$OUT/trace_replay.log" 2>&1
    echo "trace_replay exit=$?" >> "$OUT/summary.log"
    if [ "$PROBE" = probe ]; then
        bash scripts/launch_gpu_job.sh $G1 0 "$OUT/formal_probe" -- "$PY" -m revrender.train --stage formal --bundle "$B" --bundle-sha "$BS" \
            --corrected-bundle "$C" --corrected-sha "$CS" --backbone-weights "$WEIGHTS/R-50-torchvision.pkl" --gate-root "$OUT/train" \
            --output-dir "$OUT/formal_probe" > "$OUT/formal_probe.log" 2>&1 &
        local job=$!
        until [ -s "$OUT/formal_probe/curve.jsonl" ]; do sleep 10; kill -0 $job 2>/dev/null || break; done
        # stop our own probe at its first validation: the training process, found by its exact command line
        local pid; pid="$(pgrep -f "^$PY -m revrender.train --stage formal.*$OUT/formal_probe" | head -1)"
        if [ -n "$pid" ]; then kill "$pid"; fi
        wait $job 2>/dev/null
        cpu -m revrender.tools.compare_curve "$OLD/formal/curve.jsonl" "$OUT/formal_probe/curve.jsonl" > "$OUT/curve_compare.log" 2>&1
        echo "curve_compare exit=$?" >> "$OUT/summary.log"
    fi
}

lane_baselines() {   # card 2: every native baseline and Mask Transfiner through its launcher's smoke; one threshold re-run
    for m in maskrcnn pointrend condinst solov2 mask2former maskdino dimaskdino; do
        STOP_AFTER=smoke bash scripts/launch_native_arm.sh $m $G2 1 "$B" "$BS" "$C" "$CS" "$OUT/native_smoke/$m" 12000 > "$OUT/native_smoke_$m.log" 2>&1
        echo "native_smoke_$m exit=$?" >> "$OUT/summary.log"
    done
    if [ -n "${TRANSFINER_PIN:-}" ]; then
        STOP_AFTER=smoke bash scripts/launch_native_arm.sh transfiner $G2 1 "$B" "$BS" "$C" "$CS" "$OUT/native_smoke/transfiner" 12000 > "$OUT/native_smoke_transfiner.log" 2>&1
        echo "native_smoke_transfiner exit=$?" >> "$OUT/summary.log"
    fi
    bash scripts/launch_gpu_job.sh $G2 1 "$OUT/threshold_maskrcnn" -- "$NATIVE_PY" -m revrender.baselines.native.threshold --method maskrcnn \
        --freeze "$OLD_NATIVE/maskrcnn/freeze.json" --bundle "$B" --bundle-sha "$BS" --upstream-root "$UPSTREAM_V1" --weights-dir "$WEIGHTS" \
        --output-dir "$OUT/threshold_maskrcnn" > "$OUT/threshold_maskrcnn.log" 2>&1
    echo "threshold_maskrcnn exit=$?" >> "$OUT/summary.log"
}

lane_components() {  # card 0, after inference: the four revision-specific methods through their --smoke
    comp() { local name=$1; shift
        bash scripts/launch_gpu_job.sh $G0 0 "$OUT/comp_$name" -- "$PY" -m "$@" --smoke --workers 8 --output-dir "$OUT/comp_$name" > "$OUT/comp_$name.log" 2>&1
        echo "comp_$name exit=$?" >> "$OUT/summary.log"; }
    comp component_cls revrender.baselines.component_cls --bundle "$B" --bundle-sha "$BS" --corrected "$C" --corrected-sha "$CS" \
        --resnet34 "${RESNET34:-/home/zsh/hme/work/revision_gold105_seed42_v1/weights/resnet34-b627a593.pth}"
    comp mathscrub revrender.baselines.mathscrub --bundle "$B" --bundle-sha "$BS" --corrected "$C" --corrected-sha "$CS"
    comp shivakumara revrender.baselines.paper --arm shivakumara2022 --bundle "$B" --bundle-sha "$BS" --corrected "$C" --corrected-sha "$CS" \
        --densenet121-weights "${DENSENET121:-/home/zsh/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth}"
    comp chaudhuri revrender.baselines.paper --arm chaudhuri2017 --bundle "$B" --bundle-sha "$BS" --corrected "$C" --corrected-sha "$CS"
    comp removal revrender.baselines.removal --bundle "$B" --bundle-sha "$BS" \
        --release "${HEIL_RELEASE:-/home/zsh/hme/artifacts/heil_cyclegan_checkpoints_v1/checkpoints.zip}" \
        --heil-code "${HEIL_CODE:-/home/zsh/hme/artifacts/public_strikeout_sources_v2/code/strikethrough-removal-cyclegans/strikethrough_removal}"
}

lane_cpu() {         # the test suite, the baselines re-scored, found/drawn, the writer bootstrap
    REVRENDER_RESNET50="$WEIGHTS/R-50-torchvision.pkl" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 cpu -m pytest tests -q > "$OUT/pytest.log" 2>&1
    echo "pytest exit=$?" >> "$OUT/summary.log"
    cpu -m revrender.eval.report --bundle "$B" --bundle-sha "$BS" $FREEZES --reference-ap "$REF/ap883.json" \
        --reference-table "$REF/main_table.json" > "$OUT/rescore.log" 2>&1
    echo "rescore exit=$?" >> "$OUT/summary.log"
    cpu -m revrender.eval.fates --bundle "$B" --bundle-sha "$BS" $FREEZES --output "$OUT/fates.json" > "$OUT/fates.log" 2>&1
    echo "fates exit=$?" >> "$OUT/summary.log"
    cpu -m revrender.eval.bootstrap --bundle "$B" --bundle-sha "$BS" $(for m in $MODELS; do echo --variant "$m=freeze:$REF/freeze_$m.json"; done) \
        --pair dimaskdino-maskdino --pair pointrend-maskrcnn --pair rvr-condinst --pair rvr-dimaskdino --pair rvr-mask2former \
        --pair rvr-maskdino --pair rvr-maskrcnn --pair rvr-pointrend --pair rvr-solov2 --resamples 5000 --workers 8 \
        --output "$OUT/bootstrap.json" > "$OUT/bootstrap.log" 2>&1
    echo "bootstrap exit=$?" >> "$OUT/summary.log"
}

( lane_inference; lane_components ) &
lane_training &
lane_baselines &
lane_cpu &
wait

# ---- the comparisons that need more than an exit code
cpu - "$OUT" "$REF" "$OLD" "$OLD_NATIVE" >> "$OUT/summary.log" <<'EOF'
import json, sys
from pathlib import Path
out, ref, old, old_native = (Path(a) for a in sys.argv[1:5])
load = lambda p: json.loads(Path(p).read_text())
checks = {}
def last(path):
    return json.loads(Path(path).read_text().strip().splitlines()[-1])
checks['inference_report_all_checks'] = last(out/'infer_report.log').get('all_checks_pass')
checks['rescore_all_checks'] = last(out/'rescore.log').get('all_checks_pass')
a, b = load(old/'freeze.json'), load(out/'freeze_rvr_rederived.json')
checks['freeze_rederived'] = all(a['validation'][k] == b['validation'][k] for k in ('own_best', 'by_mask_threshold', 'tp_within_fp_budget')) \
    and all(a[k] == b[k] for k in ('mask_threshold', 'score_threshold', 'nms', 'chosen_step', 'checkpoint_sha256'))
reference = load(ref/'fsr_analysis.json')['variants']
fates = load(out/'fates.json')['models']
checks['fates_equal_paper1'] = all(reference[m]['errors']['gold'] == fates[m]['gold'] for m in fates)
boot, ref_boot = load(out/'bootstrap.json'), load(ref/'bootstrap887.json')
checks['bootstrap_equal_paper1'] = all(
    abs(boot['pairs'][p][m][f] - ref_boot['pairs'][p][m][f]) < 1e-12 if f != 'ci95'
    else all(abs(x - y) < 1e-12 for x, y in zip(boot['pairs'][p][m][f], ref_boot['pairs'][p][m][f]))
    for p in ref_boot['pairs'] for m in ref_boot['pairs'][p] for f in ('difference', 'ci95', 'share_not_above_zero'))
new, paper = load(out/'threshold_maskrcnn'/'freeze.json'), load(ref/'freeze_maskrcnn.json')
checks['threshold_rerun_equals_paper1'] = all(new['validation'][k] == paper['validation'][k] for k in
                                              ('own_best', 'by_mask_threshold', 'tp_within_fp_budget')) \
    and all(new[k] == paper[k] for k in ('mask_threshold', 'score_threshold', 'nms'))
print('CHECKS ' + json.dumps(checks, sort_keys=True))
EOF
echo "---- summary ($OUT/summary.log)"; cat "$OUT/summary.log"
