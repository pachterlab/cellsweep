#!/bin/bash
# Run figure jobs one after another: bash scripts/paper_figures/queue.sh [JOBFILE]   (default: jobs.txt next to this script)
#
# Each line of JOBFILE is one job (# starts a comment):
#   NOTEBOOK TAG [--set name=expr ...] [--cfg key=expr ...]   a notebook in notebooks/, executed by run_nb.py
#   !TAG command ...                                          a shell command, run from the repo root
# Lines are expanded by the shell, so they can use the variables below.
#
# Settings (override from the environment, e.g. THREADS=32 OVERWRITE=True bash scripts/paper_figures/queue.sh):
#   PYTHON      python with cellsweep[analysis] installed                     (default: python on PATH)
#   THREADS     threads for CellSweep and other tools                          (default 16)
#   DOCKER      container runtime for the R tools: docker or podman            (default podman)
#   SCAR_ENV    conda env with scAR, used only when a notebook reruns scAR     (default: empty)
#   OVERWRITE   True to recompute outputs that already exist, e.g. after a model change (default False)
#
# Per-job logs go to notebooks/output/paper_figures/logs/TAG.log, executed notebooks to
# notebooks/output/paper_figures/executed/TAG.ipynb, and one summary line per job to logs/queue_<JOBFILE>.log.
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
jobfile="$(cd "$(dirname "${1:-$here/jobs.txt}")" && pwd)/$(basename "${1:-$here/jobs.txt}")"

export PYTHON="${PYTHON:-python}"
export THREADS="${THREADS:-16}"
export DOCKER="${DOCKER:-podman}"
export SCAR_ENV="${SCAR_ENV:-}"
export OVERWRITE="${OVERWRITE:-False}"
# the scripts take --overwrite as a flag rather than a value
if [[ "$OVERWRITE" == "True" ]]; then export OVERWRITE_FLAG="--overwrite"; else export OVERWRITE_FLAG=""; fi

logs="$repo/notebooks/output/paper_figures/logs"
mkdir -p "$logs"
summary="$logs/queue_$(basename "$jobfile" .txt).log"
cd "$repo"
while IFS= read -r line; do
    [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" == "!"* ]]; then   # "!TAG command ...": a shell command run from the repo root
        read -r tag cmd <<< "${line#!}"
        echo "$(date +%F\ %T) START $tag" >> "$summary"
        (cd "$repo" && eval "$cmd") > "$logs/$tag.log" 2>&1
        rc=$?
        echo "$(date +%F\ %T) END $tag exit=$rc $(tail -1 "$logs/$tag.log")" >> "$summary"
        continue
    fi
    eval "set -- $line"
    tag="$2"
    echo "$(date +%F\ %T) START $tag" >> "$summary"
    "$PYTHON" "$here/run_nb.py" "$@" > "$logs/$tag.log" 2>&1
    rc=$?
    echo "$(date +%F\ %T) END $tag exit=$rc $(tail -1 "$logs/$tag.log")" >> "$summary"
done < "$jobfile"
echo "$(date +%F\ %T) QUEUE DONE" >> "$summary"
