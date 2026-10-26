#!/usr/bin/env bash
# A batch of training questions for the struck-text page (user 2026-10-23: pixels and struck text in one pass, as the test page):
# questions of paper 1's training writers not annotated before (data.annotation_set: one per writer first, every exam question in
# turn; leaves out earlier batches and every question an export shows worked on), the pixels prefilled by the NAFNet the user chose
# (pixel_library infer --restorer), the struck text drafted by Qwen3-VL-32B on the prefill's erased crops, the text-mode page.
#   build_training_batch.sh BATCH LINES NAFNET_RUN NAFNET_STEP     e.g. 2 900 paper2_clean_prefill_model_main step_8372
# Output paper2_training_batch<BATCH>_{selection,pack,export,crops,erased,read32b,page}; the page is site/标注.html (inline KaTeX locally
# with struck_text update --katex).  Whichever cards have room (_common.sh open_card: card 3 only while nobody else is on it).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_clean_source
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BATCH="$1"; LINES="$2"; RUN="$3"; STEP="$4"
O=/home/zsh/hme/outputs; T=$O/paper2_training_batch$BATCH; FT=$O/paper2_pixel_finetune_inputs20261007; LAB=$O/paper2_annotation_labels
Q=/home/zsh/hme/code_roi725_v1/outputs
index() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="$1" '$2==u{print $1}'; }
test ! -e "${T}_selection"
trap 'touch "$T.FAILED"' ERR
earlier=(); for s in "$O"/paper2_training_batch*_selection/selection.json; do [ -e "$s" ] && earlier+=(--exclude "$s"); done
exports=(); for e in "$O/paper2_pixel_round_struck20261012_inputs/training_annotations.json" "$FT/annotations_0.json" "$FT/annotations_1.json" \
                    "$LAB/全量划线标注_2026-10-11.json" "$LAB/划线精细标注_paper2-struck-pixels-5a688fece9ad_2026-10-11.json" \
                    "$O"/paper2_training_batch*_returned/export.json; do [ -e "$e" ] && exports+=(--exclude-export "$e"); done
"$PY" -m revrender.data.annotation_set --catalogue /home/zsh/hme/code_358_spatial_v1/gold/line_edit_all5002_policy_decisions_20260814_v1/canonical/exact_crops_46902.jsonl \
    --mapping "$FT/index_manifest.json" --same-student "$FT/same_student_pairs.json" --lines "train=$LINES" \
    --pilot 0 --exclude "$O/paper2_annotation_labels_allowed20261021/selection.json" "${earlier[@]}" "${exports[@]}" --output-dir "${T}_selection"
"$PY" -c 'import json,sys; print("\n".join(p["image_id"] for p in json.load(open(sys.argv[1]))["pages"]))' "${T}_selection/selection.json" > "${T}_selection/ids.txt"
until card=$(open_card 8000); do sleep 30; done; [ "$card" != "$CARD3" ] || export ALLOW_GPU3=1
bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$(numa_of "$card")" "${T}_pack" -- "$PY" -m revrender.tools.annotate.pixel_library infer --restorer "$O/$RUN" \
    --restorer-checkpoint "$STEP" --upstream "$O/paper2_restoration_sources20261008" --pages "$O/paper2_pages_clean_unlabeled3976" \
    --detector "$O/paper2_pages_clean_run_unlabeled3976_tight" --ids "${T}_selection/ids.txt" --mapping "$FT/index_manifest.json" \
    --same-student "$FT/same_student_pairs.json" --validation "$O/paper2_fixed_validation20261007/validation_native.json" \
    --name "paper2-training-batch$BATCH" --output-dir "${T}_pack"
"$PY" -m revrender.tools.annotate.pixel_library prefill-export --pack "${T}_pack" --output-dir "${T}_export"
"$PY" -m revrender.tools.annotate.struck_text crops --export "${T}_export/export.json" --review "${T}_pack" --pages /home/zsh/hme/data/unlabeled_v1/images \
    --split train --unconfirmed --output-dir "${T}_crops"
"$PY" -m revrender.tools.annotate.struck_text erased --crops "${T}_crops" --export "${T}_export/export.json" --review "${T}_pack" --output-dir "${T}_erased"
until a=$(open_card 21000) && b=$(open_card 21000 "$a"); do sleep 30; done; case "$a,$b" in *314fdf43*) export ALLOW_GPU3=1 ;; esac
bash "$ROOT/scripts/launch_gpu_job.sh" "$a,$b" "$(numa_of "$a")" "${T}_read32b" -- env -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$(index "$a"),$(index "$b")" "$PY" -m revrender.recognize_lines --pack "${T}_erased" --conditions erased_neutral \
    --model /home/zsh/hme/models/Qwen3-VL-32B-Instruct-FP8 --tp 2 --gpu-memory .85 --output-dir "${T}_read32b"
"$PY" -m revrender.tools.annotate.struck_text page --crops "${T}_crops" --export "${T}_export/export.json" --readings "${T}_read32b/predictions.jsonl" \
    --condition erased_neutral --questions "$Q/line_edit_question_answer_authority_20260814_v1/questions_with_answers.jsonl" \
    --answer-groups "$Q/line_edit_response_index_20260814_v1/answer_groups.jsonl" --output-dir "${T}_page"
touch "$T.COMPLETE"
