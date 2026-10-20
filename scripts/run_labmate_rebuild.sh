#!/usr/bin/env bash
# Rebuild everything a labmate's data reached (user 2026-10-21), unattended and resumable.
#   run_labmate_rebuild.sh            (cards 0, 1, 2; card 3 only when nobody else is on it)
# 0. Copy the frozen validation out, write filtered copies of the labels that named her pages, and move every output holding her
#    pages, and every model trained on them, to paper2_quarantine_labmate20261021 (nothing is deleted; QUARANTINE lists them).
# 1. Card 1: print templates, paper 1's retrained RevRender (945, no labmate page) on the unlabeled pages, tighten; paper 1's gold
#    put back on allowed pages, the two stroke banks, the synthetic line and box sets, the restoration pairs; then our model:
#    synthetic lines -> synthetic boxes -> the step chosen on the 41 validation strikes -> the 201 human regions -> the full
#    library's prefill.  Cards 0 and 2: the seven restoration baselines on the new pairs.  Card 1 afterwards: the recognition
#    pilot (line packs, VLM readings, token hiding, reader LoRAs and their evaluations) replayed on clean inputs.
# 2. Last, on cards 0-2: the struck-content diagnostic study with the new detector and box remover, and the 32B line reading.
# Every new output is paper2_clean_<old name>; a step whose marker exists in paper2_rebuild20261021 is skipped.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs; Q=$O/paper2_quarantine_labmate20261021; R=$O/paper2_rebuild20261021; N=$O/paper2_clean
UNL=/home/zsh/hme/data/unlabeled_v1/images; CLEAN=$O/paper2_pages_clean_unlabeled3976
P1=/home/zsh/hme/work/revision_final_runs945_v1/rvr; P1_SHA=$(sha256sum "$P1/formal/best.pt" | cut -d' ' -f1)
MAP=$O/paper2_splits/index_manifest.json; PAIRS=$O/paper2_splits/same_student_pairs.json
LAB=$O/paper2_annotation_labels_allowed20261021; VAL=$O/paper2_fixed_validation20261007
C0=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; C1=GPU-d0bcc564-5bd9-b943-f995-8536415088e3; C2=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0
M=/home/zsh/hme/models
mkdir -p "$R"
stamp() { echo "$* $(date -u +%FT%TZ)"; }
done_() { [ -e "$R/$1.done" ]; }
mark() { touch "$R/$1.done"; stamp "done $1"; }
gpu() { bash "$ROOT/scripts/launch_gpu_job.sh" "$@"; }
sha() { sha256sum "$1/run_manifest.json" | cut -d' ' -f1; }
# every old output maps to paper2_clean_<suffix>, in the recorded paths of a replayed command
# (the explicit ones first: among keys of one length the first wins)
MAPS=(--map "$O/paper2_reader_corpus_main=${N}_reader_corpus" --map "$O/paper2_reader_corpus_stack=${N}_reader_corpus"
      --map "$O/paper2_reader_corpus=${N}_reader_corpus" --map "$O/paper2_prior_annotations_in=${N}_prior_annotations"
      --map "$O/paper2_pixel_finetune_set20261007/validation_native.json=$VAL/validation_native.json"
      --map "/home/zsh/hme/work/revision_tune880_v1/control=$P1")
for old in $(cat "$R/quarantine.txt" 2>/dev/null) paper2_pages_clean_run_unlabeled3976 paper2_pages_clean_run_unlabeled3976_tight; do
    MAPS+=(--map "$O/$old=${N}_${old#paper2_}"); done
replay() {   # NAME OLD NEW [--set ...]: replay OLD's recorded command into NEW (the old one now sits in the quarantine if it held her pages)
    local name="$1" old="$2" new="$3"; shift 3
    [ -d "$Q/$(basename "$old")" ] && old="$Q/$(basename "$old")"
    "$PY" -m revrender.tools.replay --from "$old" --to "$new" "${MAPS[@]}" "$@" > "$R/$name.log" 2>&1
}
replay_gpu() {   # NAME CARD NUMA INDEX OLD NEW [--set ...]
    local name="$1" card="$2" numa="$3" index="$4" old="$5" new="$6"; shift 6
    [ -d "$Q/$(basename "$old")" ] && old="$Q/$(basename "$old")"
    gpu "$card" "$numa" "$new" -- env -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES="$index" "$PY" -m revrender.tools.replay --from "$old" --to "$new" "${MAPS[@]}" "$@" > "$R/$name.log" 2>&1
}
trap 'stamp "rebuild failed at line $LINENO"; touch "$R/FAILED"' ERR

# ---- 0. set the labmate's data aside ---------------------------------------------------------------------------------------------
if ! done_ quarantine; then
    test -s "$R/quarantine.txt"            # the folders to move, from the scan of 2026-10-21 (98 holding her pages, 40 trained on them)
    if [ ! -e "$VAL" ]; then
        mkdir "$VAL"; cp "$O/paper2_pixel_finetune_set20261007/validation_native.json" "$O/paper2_pixel_finetune_set20261007/validation_native.npz" "$VAL/"
        cmp "$VAL/validation_native.npz" "$O/paper2_pixel_finetune_set20261007/validation_native.npz"
    fi
    "$PY" - "$O" "$LAB" <<'PY'
import json, sys
from pathlib import Path
from revrender.data.allowed import allowed
from revrender.util import digest, make_output_dir, write_json, write_manifest
O, LAB = Path(sys.argv[1]), Path(sys.argv[2])
if not LAB.exists():
    out = make_output_dir(LAB); src = O/'paper2_annotation_labels'; inputs = {}
    sel = json.loads((src/'selection.json').read_text())
    write_json(out/'selection.json', {**sel, 'pages': [p for p in sel['pages'] if allowed(p['image_id'])]})
    for name in ('涂改标注_paper2-revision-boxes-test_2026-10-05-04-28.json',):
        d = json.loads((src/name).read_text())
        write_json(out/name, {**d, 'pages': {k: v for k, v in d['pages'].items() if allowed(k)}})
    user = O/'paper2_test_text20261018_inputs'/'user_export_20261018.json'
    d = json.loads(user.read_text())
    write_json(out/'test_export_20261018.json', {**d, 'pages': {k: v for k, v in d['pages'].items() if allowed(k)}})
    write_manifest(out, inputs={str(p): digest(p) for p in (src/'selection.json', src/name, user)},
                   note='copies without a labmate page (user 2026-10-21); the originals are in the quarantine')
PY
    mkdir -p "$Q/paper2_annotation_labels" "$Q/paper2_test_text20261018_inputs" "$Q/paper2_pixel_batch80_inputs20261007"
    for f in selection.json test_ids.txt 涂改标注_paper2-revision-boxes-test_2026-10-05-04-28.json; do
        [ ! -e "$O/paper2_annotation_labels/$f" ] || mv "$O/paper2_annotation_labels/$f" "$Q/paper2_annotation_labels/"; done
    [ ! -e "$O/paper2_test_text20261018_inputs/user_export_20261018.json" ] || mv "$O/paper2_test_text20261018_inputs/user_export_20261018.json" "$Q/paper2_test_text20261018_inputs/"
    [ ! -e "$O/paper2_pixel_batch80_inputs20261007/selection.json" ] || mv "$O/paper2_pixel_batch80_inputs20261007/selection.json" "$Q/paper2_pixel_batch80_inputs20261007/"
    while read -r d; do
        [ -n "$d" ] || continue
        [ ! -e "$O/$d" ] || mv "$O/$d" "$Q/"
        for f in "$O/$d".*; do [ ! -e "$f" ] || mv "$f" "$Q/"; done
    done < "$R/quarantine.txt"
    "$PY" -m revrender.data.allowed scan "$Q/paper2_annotation_labels" "$LAB" "$VAL" > "$R/quarantine_check.log"
    mark quarantine
fi

# ---- 1. card 1: detections, data, our model; cards 0 and 2: baselines; card 1 after: the recognition pilot --------------------------
lane_ours() {
    if ! done_ templates; then mkdir -p "${N}_page_templates"; "$PY" -m revrender.data.handwriting --pages "$UNL" --output "${N}_page_templates/templates.npz" > "$R/templates.log" 2>&1; mark templates; fi
    if ! done_ detect; then
        replay_gpu detect "$C1" 0 1 "$O/paper2_pages_clean_run_unlabeled3976" "${N}_pages_clean_run_unlabeled3976" \
            --set "--checkpoint=$P1/formal/best.pt" --set "--checkpoint-sha=$P1_SHA" --set "--freeze=$P1/freeze.json"
        replay tighten "$O/paper2_pages_clean_run_unlabeled3976_tight" "${N}_pages_clean_run_unlabeled3976_tight"; mark detect; fi
    if ! done_ gold; then
        replay gold "$O/paper2_gold_pages" "${N}_gold_pages"
        replay gold_tight "$O/paper2_gold_pages_tight_connected" "${N}_gold_pages_tight_connected"
        replay bank "$O/paper2_strike_bank" "${N}_strike_bank"
        replay bank_strokes "$O/paper2_strike_bank_strokes" "${N}_strike_bank_strokes"; mark gold; fi
    if ! done_ synthetic; then
        replay synthetic_strokes "$O/paper2_synthetic_strokes" "${N}_synthetic_strokes"
        replay synthetic_boxes "$O/paper2_synthetic_boxes" "${N}_synthetic_boxes"; mark synthetic; fi
    if ! done_ pairs; then
        replay pairs "$O/paper2_restoration_pairs20261007_v3" "${N}_restoration_pairs20261007_v3"
        replay pairs_darken "$O/paper2_restoration_pairs20261007_v4" "${N}_restoration_pairs20261007_v4"; mark pairs; fi
    if ! done_ control; then
        SYNTHETIC=1 MAPPING=$MAP RECIPE=configs/revrender_strokes.toml bash "$ROOT/scripts/launch_train_arm.sh" control "$C1" 0 \
            "${N}_synthetic_strokes" "$(sha "${N}_synthetic_strokes")" - - "${N}_strokes_control" > "$R/control.log" 2>&1; mark control; fi
    if ! done_ boxes; then
        last=$(ls "${N}_strokes_control/formal"/step_*.pt | sort -t_ -k3 -n | tail -1)
        SYNTHETIC=1 MAPPING=$MAP SAME_STUDENT=$PAIRS INIT=$last RECIPE=configs/revrender_strokes_boxes.toml bash "$ROOT/scripts/launch_train_arm.sh" \
            control "$C1" 0 "${N}_synthetic_boxes" "$(sha "${N}_synthetic_boxes")" - - "${N}_strokes_boxes" > "$R/boxes.log" 2>&1; mark boxes; fi
    if ! done_ select_boxes; then   # the rule of 2026-10-05: the highest read-back of Qwen3-VL-8B on the 41 validation strikes, each box's best mask
        V="${N}_strokes_boxes_val_top1"; mkdir -p "$V"; MODELS=()
        for ck in "${N}_strokes_boxes/formal"/step_*.pt; do
            s=$(basename "$ck" .pt)
            [ -e "$V/$s/complete.json" ] || [ -e "$V/$s/run_manifest.json" ] || gpu "$C1" 0 "$V/$s" -- "$PY" -m revrender.box_remover --checkpoint "$ck" \
                --point 0.4 0.5 0 --labels "$O/paper2_annotation_labels/涂改标注_paper2-revision-boxes-val_2026-10-05-08-08.json" --selection "$LAB/selection.json" \
                --split val --top 1 --pages "$UNL" --output-dir "$V/$s" >> "$R/select_boxes.log" 2>&1
            MODELS+=(--model "t$s=$V/$s")
        done
        "$PY" -m revrender.eval.revision_boxes boxes --labels "$O/paper2_annotation_labels/涂改标注_paper2-revision-boxes-val_2026-10-05-08-08.json" \
            --selection "$LAB/selection.json" --split val "${MODELS[@]}" --pages "$UNL" --cuts 0.0 0.1 0.39 --output-dir "${N}_strokes_val_boxes_top1" >> "$R/select_boxes.log" 2>&1
        gpu "$C1" 0 "${N}_strokes_val_boxes_top1_qwen8b" -- env -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS CUDA_VISIBLE_DEVICES=1 \
            "$PY" -m revrender.recognize_lines --pack "${N}_strokes_val_boxes_top1/pack" --model $M/Qwen3-VL-8B-Instruct --conditions erased_neutral \
            --gpu-memory .85 --output-dir "${N}_strokes_val_boxes_top1_qwen8b" >> "$R/select_boxes.log" 2>&1
        "$PY" -m revrender.eval.revision_boxes read --pack "${N}_strokes_val_boxes_top1/pack" --readings "${N}_strokes_val_boxes_top1_qwen8b/predictions.jsonl" \
            --output "${N}_strokes_val_boxes_top1/read_qwen8b.json" >> "$R/select_boxes.log" 2>&1
        "$PY" - "${N}_strokes_val_boxes_top1/read_qwen8b.json" "${N}_strokes_boxes/formal" "$R/selected_box_step.json" <<'PY'
import json, sys
held = json.load(open(sys.argv[1]))['held']
scores = {k.split('/')[0][1:]: v['held'] for k, v in held.items() if k.endswith('/erased_neutral/line') and k.startswith('tstep_')}
best = max(scores, key=lambda s: (scores[s], -int(s.split('_')[1])))          # highest read-back; the earlier step on a tie
json.dump({'rule': 'highest Qwen3-VL-8B read-back on the 41 validation strikes (each box its best mask); earlier step on a tie',
           'scores': scores, 'selected': best, 'checkpoint': f'{sys.argv[2]}/{best}.pt'}, open(sys.argv[3], 'x'), indent=1)
print(best, scores[best])
PY
        mark select_boxes; fi
    BOX=$("$PY" -c "import json; print(json.load(open('$R/selected_box_step.json'))['checkpoint'])")
    if ! done_ finetune; then
        [ -e "${N}_pixel_finetune_set" ] || "$PY" -m revrender.data.review_set \
            --review-export "$O/paper2_pixel_round_struck20261012_inputs/training_annotations.json" --pages "$O/paper2_full_review20261010/images" \
            --mapping "$O/paper2_pixel_finetune_inputs20261007/index_manifest.json" --same-student "$O/paper2_pixel_finetune_inputs20261007/same_student_pairs.json" \
            --replay "${N}_synthetic_boxes" --fixed-validation "$VAL" --output-dir "${N}_pixel_finetune_set" > "$R/finetune_set.log" 2>&1
        bash "$ROOT/scripts/run_review_round.sh" "${N}_pixel_finetune_set" "$BOX" "$C1" 0 "${N}_pixel_finetune" > "$R/finetune.log" 2>&1; mark finetune; fi
    if ! done_ prefill; then   # the full library's regions as before, prefilled by the new model (the browser keeps every human edit)
        gpu "$C1" 0 "${N}_full_review_prefill" -- "$PY" -m revrender.tools.annotate.pixel_library infer --selected "${N}_pixel_finetune/eval/summary.json" \
            --pages "$CLEAN" --detector "$O/paper2_pages_clean_run_unlabeled3976_tight" --mapping "$O/paper2_pixel_finetune_inputs20261007/index_manifest.json" \
            --same-student "$O/paper2_pixel_finetune_inputs20261007/same_student_pairs.json" --validation "$VAL/validation_native.json" \
            --review-export "$O/paper2_pixel_finetune_inputs20261007/annotations_0.json" --review-export "$O/paper2_pixel_finetune_inputs20261007/annotations_1.json" \
            --name paper2-full-review --output-dir "${N}_full_review_prefill" > "$R/prefill.log" 2>&1
        "$PY" -m revrender.tools.annotate.pixel_library bundle --pack "${N}_full_review_prefill" --pages-per-part 20 --output-dir "${N}_full_review_prefill_site" >> "$R/prefill.log" 2>&1
        mark prefill; fi
    # the recognition pilot of 2026-10-03/04 on clean inputs, one card
    for item in line_pack_s0 line_pack_s0b; do
        done_ "$item" || { replay_gpu "$item" "$C1" 0 1 "$O/paper2_$item" "${N}_$item" --set "--checkpoint=$P1/formal/best.pt" \
            --set "--checkpoint-sha=$P1_SHA" --set "--freeze=$P1/freeze.json"; mark "$item"; }; done
    for item in s0_qwen3vl8b s0b_qwen25vl7b s0b_qwen3vl2b s0b_qwen3vl4b s0b_qwen3vl8b tokens_qwen3vl2b tokens_qwen3vl8b \
                reader_qwen3vl8b_strikes reader_qwen3vl8b_clean reader_qwen3vl2b_strikes reader_qwen3vl2b_clean reader_qwen3vl2b_clean_matched \
                reader_qwen3vl2b_clean_whole reader_qwen3vl2b_all4 reader_qwen3vl2b_blank \
                readeval_8b_base readeval_8b_clean readeval_8b_strikes readeval_2b_base readeval_2b_clean readeval_2b_strikes \
                readeval_2b_clean_matched readeval_2b_clean_whole readeval_2b_all4 readeval_2b_blank; do
        done_ "$item" || { replay_gpu "$item" "$C1" 0 1 "$O/paper2_$item" "${N}_$item"; mark "$item"; }; done
    for item in tokens_qwen3vl2b_sweep tokens_qwen3vl8b_sweep; do   # one recorded command per setting
        if ! done_ "$item"; then
            src="$Q/paper2_$item"; mkdir -p "${N}_$item"
            for sub in "$src"/*/; do s=$(basename "$sub"); [ -e "${N}_$item/$s/complete.json" ] || replay_gpu "${item}_$s" "$C1" 0 1 "$sub" "${N}_$item/$s"; done
            mark "$item"; fi; done
}
lane_baselines() {   # LANE CARD MODELS...: the restoration baselines on the new pairs (protocol unchanged)
    local lane="$1" card="$2"; shift 2
    until done_ pairs; do sleep 300; done
    done_ "baselines_$lane" && return 0
    bash "$ROOT/scripts/run_restoration_suite.sh" "$lane" "${N}_restoration_pairs20261007_v4" "$O/paper2_restoration_sources20261008" \
        "${N}_restoration_suite/lane$lane" formal "$card" "$@" > "$R/baselines_$lane.log" 2>&1
    mark "baselines_$lane"
}
mkdir -p "${N}_restoration_suite"
( lane_ours ) > "$R/lane_ours.log" 2>&1 & L1=$!
( lane_baselines 0 0 COR-Generator Restormer Strikethrough-CycleGAN DocDiff ) > "$R/lane_b0.log" 2>&1 & L0=$!
( lane_baselines 2 2 NAFNet DE-GAN
  if ! done_ stainrestorer; then
      bash "$ROOT/scripts/run_restoration_suite.sh" 2 "${N}_restoration_pairs20261007_v4" "$O/paper2_stainrestorer_resources20261009" \
          "${N}_stainrestorer" formal 2 StainRestorer > "$R/stainrestorer.log" 2>&1; mark stainrestorer; fi ) > "$R/lane_b2.log" 2>&1 & L2=$!
wait $L1; wait $L0; wait $L2

# ---- 2. two-card work last: the diagnostic study with the rebuilt models, and the 32B line reading ----------------------------------
BOX=$("$PY" -c "import json; print(json.load(open('$R/selected_box_step.json'))['checkpoint'])")
if ! done_ s0b_qwen3vl32b; then replay_gpu s0b_qwen3vl32b "$C1,$C2" 0 1,2 "$O/paper2_s0b_qwen3vl32b" "${N}_s0b_qwen3vl32b"; mark s0b_qwen3vl32b; fi
if ! done_ diagnostics; then
    C1=$C1 I1=1 C2=$C0 I2=0 PAIR="$C0,$C2" PAIR_INDICES=0,2 BUILD_ARGS="--detector ${N}_pages_clean_run_unlabeled3976_tight --box-remover $BOX" \
        bash "$ROOT/scripts/run_struck_diagnostics.sh" "${N}_struck_diag" > "$R/diagnostics.log" 2>&1
    mark diagnostics; fi
touch "$R/COMPLETE"; stamp "rebuild complete"
