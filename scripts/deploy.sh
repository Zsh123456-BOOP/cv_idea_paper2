#!/usr/bin/env bash
# Deploy a committed snapshot of this repository to the server: /home/zsh/hme/code_paper2_<commit>.
#   scripts/deploy.sh [HOST]        (default 4090x4; run from a clean tree on the commit to deploy)
# The snapshot is `git archive` (tracked files only: no data, no weights) plus deployment_origin.json naming the commit.
# An existing snapshot directory is never touched; a new commit gets a new directory.
set -Eeuo pipefail
HOST="${1:-4090x4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" || { echo 'source dirty: commit first'; exit 2; }
COMMIT="$(git -C "$ROOT" rev-parse --short=8 HEAD)"
FULL="$(git -C "$ROOT" rev-parse HEAD)"
DEST="/home/zsh/hme/code_paper2_${COMMIT}"
ssh "$HOST" "test ! -e '$DEST'" || { echo "snapshot exists: $DEST (reuse it)"; exit 0; }
git -C "$ROOT" archive --format=tar HEAD | ssh "$HOST" "mkdir '$DEST' && tar -x -C '$DEST'"
ARCHIVE_SHA="$(git -C "$ROOT" archive --format=tar HEAD | shasum -a 256 | cut -d' ' -f1)"
ssh "$HOST" "printf '{\n  \"source_commit\": \"%s\",\n  \"archive_sha256\": \"%s\"\n}\n' '$FULL' '$ARCHIVE_SHA' > '$DEST/deployment_origin.json'"
echo "$DEST"
