# Steps the restorer experiments share (run_annotation_value.sh, run_learning_curve.sh, run_realistic_synthesis.sh); sourced after
# _common.sh, with O, VAL, UP, STAIN_UP, C1, C2, C3 set.  Versions are chosen by eval.unified_restoration choose (read-back on the 41
# validation strikes among the versions whose pixel F1 on the 16 fixed regions improved, and the last).
second_card() {   # card 2 when it has room, else card 3 when nobody is on it -> its uuid
    while true; do
        [ "$(nvidia-smi -i "$C2" --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 15000 ] || { echo "$C2"; return; }
        [ -n "$(nvidia-smi -i "$C3" --query-compute-apps=pid --format=csv,noheader)" ] || { echo "$C3"; return; }
        sleep 60
    done
}
evaluate_versions() {   # LIST OUT LOG: every version of LIST (eval.unified_restoration candidates) evaluated together on validation, then chosen
    local list="$1" out="$2" log="$3"
    [ -e "$out/COMPLETED" ] || MODELS="$list" GPU_A=$C1 GPU_B=$(second_card) bash "$ROOT/scripts/eval_unified_restoration.sh" "$out" >> "$log" 2>&1
    [ -e "$out/chosen/selection.json" ] || "$PY" -m revrender.eval.unified_restoration choose --evaluation "$out" --splits ${CHOOSE_SPLITS:-val} \
        --output-dir "$out/chosen" >> "$log" 2>&1
}
chosen_of() {   # SELECTION FAMILY: the chosen version's weights (a box remover's .pt, a restorer's step directory)
    "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1]))[sys.argv[2]]["spec"]; print(s["checkpoint"] if s["type"]=="own" else s["run"]+"/"+s["checkpoint"])' "$1/selection.json" "$2"
}
finetune_restorer() {   # MODEL CARD NUMA INIT DATA OUT [BUDGET]: a restorer trained on DATA from INIT ('-': its own random start) with a
    # budget of configs/restoration.toml (default [finetune] from INIT)
    local model="$1" card="$2" numa="$3" init="$4" data="$5" out="$6" budget="${7:-}" runtime=$PY upstream=$UP start=()
    [ "$init" = - ] || start=(--init "$init"); [ -z "$budget" ] || start+=(--budget "$budget")
    [ ! -e "$out/selection.json" ] || return 0
    [ "$model" != StainRestorer ] || upstream=$STAIN_UP
    (
        export PYTHONPATH="$ROOT"; unset LD_LIBRARY_PATH
        if [ "$model" = DE-GAN ]; then
            runtime=/home/zsh/hme/envs/paper2_restoration_tf/bin/python; export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
            for d in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; done
        elif [ "$model" = StainRestorer ]; then export PYTHONPATH="$ROOT:/home/zsh/hme/envs/paper2_stain_deps"; fi
        idle=0; if [ "$card" = "$C3" ]; then idle=1; export ALLOW_GPU3=1; fi
        wait_for_gpu_memory "$card" 20000 "$idle"
        bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$out" -- "$runtime" -m revrender.baselines.restoration --model "$model" \
            --arm gold_pixels --protocol "$ROOT/configs/restoration.toml" --data "$data" --upstream "$upstream" \
            --human-validation "$VAL/validation_native.json" "${start[@]}" --output-dir "$out" > "$out.log" 2>&1 < /dev/null
    )
}
