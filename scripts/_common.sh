# Shared by the launchers (source it; do not run it).  Server conventions of this project:
#  - code runs from a committed snapshot  /home/zsh/hme/code_paper2_<commit>  (scripts/deploy.sh), never from a work tree
#  - outputs go to /home/zsh/hme/outputs/paper2_*   (create-only: a launcher refuses an existing OUT)
#  - every job holds its card's lock SHARED (flock -s), so an exclusive holder elsewhere waits for all of us;
#    at most two of our jobs per card; card 3 is not ours unless ALLOW_GPU3=1 (the user's word)
#  - never kill another user's process; never install into a shared environment
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/zsh/hme/envs/hme/bin/python}"
NATIVE_PY="${NATIVE_PY:-/home/zsh/hme/envs/native_baselines_v1/bin/python}"
LOCKS="${LOCKS:-/home/zsh/hme/work/revision_speed_gpu_locks}"
WEIGHTS="${WEIGHTS:-/home/zsh/hme/work/native_baselines_weights_v1}"
UPSTREAM_V1="${UPSTREAM_V1:-/home/zsh/hme/work/native_baselines_sources_v1}"
UPSTREAM_V2="${UPSTREAM_V2:-/home/zsh/hme/work/native_baselines_sources_v2}"
CARD3=GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b

refuse_card3() {   # $1 = GPU uuid
    case "$1" in *314fdf43*) [ "${ALLOW_GPU3:-0}" = 1 ] || { echo 'GPU 3 is not ours (ALLOW_GPU3=1 only on the user'"'"'s word)'; exit 2; } ;; esac
}

require_clean_source() {   # a deployed snapshot carries deployment_origin.json instead of .git
    [ -f "$ROOT/deployment_origin.json" ] && return 0
    test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" || { echo 'source dirty'; exit 2; }
}

hold_card() {      # $1 = GPU uuid (several joined by commas for a job that spans cards): take each card's lock shared and pin the environment
    local uuid fd
    for uuid in ${1//,/ }; do
        exec {fd}>>"$LOCKS/$uuid.lock"
        flock -s "$fd"
    done
    export CUDA_VISIBLE_DEVICES="$1" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
}

require_free_memory() {   # $1 = GPU uuid, $2 = MiB
    local free
    free="$(nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits)"
    test "$free" -ge "$2" || { echo "insufficient free GPU memory ($free < $2)"; exit 3; }
}
