#!/usr/bin/env bash
set -euo pipefail
# Dedicated environment: never installs into the existing hme environment.
TASK_BASE=/home/zsh/hme
TASK_ENV="$TASK_BASE/envs/native_baselines_v1"
TASK_SOURCES="$TASK_BASE/work/native_baselines_sources_v1"
TASK_CODE=$(cd "$(dirname "$0")/../../.." && pwd)
TASK_UV=/home/zsh/.local/bin/uv
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr
export TORCH_CUDA_ARCH_LIST=8.9 FORCE_CUDA=1 MAX_JOBS=2
export LDFLAGS=-L/usr/lib/x86_64-linux-gnu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [ ! -x "$TASK_ENV/bin/python" ]; then
    "$TASK_UV" venv --python /usr/bin/python3.12 "$TASK_ENV"
fi
"$TASK_UV" pip install --native-tls --python "$TASK_ENV/bin/python" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$TASK_UV" pip install --native-tls --python "$TASK_ENV/bin/python" 'numpy==1.26.4' scipy pandas pillow 'setuptools<76' wheel pip ninja cython fvcore iopath yacs pycocotools 'timm==1.0.20' 'omegaconf==2.3.0' 'hydra-core==1.3.2' cloudpickle termcolor tabulate tensorboard packaging matplotlib opencv-python-headless==4.10.0.84 scikit-image shapely h5py submitit einops pytest
mkdir -p "$TASK_SOURCES"
clone_pinned() {
    local name=$1 url=$2 revision=$3
    if [ ! -d "$TASK_SOURCES/$name/.git" ]; then
        git clone --depth 1 "$url" "$TASK_SOURCES/$name"
    fi
    test "$(git -C "$TASK_SOURCES/$name" rev-parse HEAD)" = "$revision"
}
clone_pinned detectron2 https://github.com/facebookresearch/detectron2.git a2f4a8771ab77e8411c26b27f24f9489a28a2453
clone_pinned Mask2Former https://github.com/facebookresearch/Mask2Former.git 9b0651c6c1d5b3af2e6da0589b719c514ec0d69a
clone_pinned MaskDINO https://github.com/IDEA-Research/MaskDINO.git 3831d8514a3728535ace8d4ecc7d28044c42dd14
clone_pinned AdelaiDet https://github.com/aim-uofa/AdelaiDet.git 5e19cb172b8363820b409ed1a2754fb19ad3acb8
PYTHONPATH="$TASK_CODE" "$TASK_ENV/bin/python" -m revrender.baselines.native.compat --upstream-root "$TASK_SOURCES"
"$TASK_UV" pip install --native-tls --python "$TASK_ENV/bin/python" --no-build-isolation -e "$TASK_SOURCES/detectron2"
for package in mask2former maskdino; do
    if [ "$package" = mask2former ]; then repo=Mask2Former; else repo=MaskDINO; fi
    (
        cd "$TASK_SOURCES/$repo/$package/modeling/pixel_decoder/ops"
        "$TASK_ENV/bin/python" setup.py build_ext --inplace
    )
done
mkdir -p "$TASK_BASE/work/native_baselines_weights_v1"
for kind in torchvision MSRA; do
    target="$TASK_BASE/work/native_baselines_weights_v1/R-50-$kind.pkl"
    if [ ! -f "$target" ]; then
        curl --fail --location --retry 3 "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/$kind/R-50.pkl" -o "$target.partial"
        mv "$target.partial" "$target"
    fi
done
"$TASK_UV" pip freeze --python "$TASK_ENV/bin/python" > "$TASK_SOURCES/environment_freeze.txt"
printf 'NATIVE_ENVIRONMENT_READY\n'
