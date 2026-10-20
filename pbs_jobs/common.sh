#!/bin/bash
# Shared setup for the PBS entry points. Source this file; use run_workflow.py locally.

ggkm_log_exit() {
    local status="$1"
    local finished_at
    finished_at="$(date +%s)"
    printf '\nFinished: %s\nExit status: %s\nElapsed seconds: %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$status" "$((finished_at - GGKM_TASK_STARTED_AT))"
}

ggkm_run_job() {
    local job_key="$1"
    local job_id="${PBS_JOBID:-manual_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    local task_suffix="${PBS_ARRAY_INDEX:+_$PBS_ARRAY_INDEX}"
    local log_stem
    local default_workers=1
    local venv_path="${GGKM_VENV-$HOME/venvs/master_thesis}"
    local -a modules

    cd "$GGKM_PROJECT_ROOT"
    export GGKM_PROJECT_ROOT="$PWD"
    if [[ ! -f scripts/run_workflow.py ]]; then
        echo "Missing scripts/run_workflow.py in GGKM_PROJECT_ROOT=$GGKM_PROJECT_ROOT" >&2
        return 1
    fi
    mkdir -p logs/pbs
    log_stem="logs/pbs/${job_key}_${job_id}${task_suffix}"
    exec > "${log_stem}.out"
    exec 2> "${log_stem}.err"
    GGKM_TASK_STARTED_AT="$(date +%s)"
    trap 'ggkm_log_exit "$?"' EXIT

    printf 'Job: %s\nJob ID: %s\nTask ID: %s\nHost: %s\nStart: %s\nWorking directory: %s\n' \
        "$job_key" "$job_id" "${PBS_ARRAY_INDEX:-none}" "$(hostname)" \
        "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$GGKM_PROJECT_ROOT"

    read -r -a modules <<< "${GGKM_MODULES-gcc/14.2.0 python/3.12.8 pcre2/10.44}"
    if (( ${#modules[@]} )); then
        if ! type module >/dev/null 2>&1; then
            echo "The module command is unavailable. Initialize environment modules or set GGKM_MODULES='' to skip them." >&2
            return 1
        fi
        module purge
        module load "${modules[@]}"
    fi

    if [[ -n "$venv_path" ]]; then
        if [[ ! -f "$venv_path/bin/activate" ]]; then
            echo "Cannot activate $venv_path. Set GGKM_VENV to the cluster environment, or to an empty string to skip activation." >&2
            return 1
        fi
        source "$venv_path/bin/activate"
    fi

    export GGKM_PYTHON="${GGKM_PYTHON:-python}"
    if ! command -v "$GGKM_PYTHON" >/dev/null 2>&1; then
        echo "Python interpreter not found: $GGKM_PYTHON" >&2
        return 1
    fi
    export R_HOME="${GGKM_R_HOME-$HOME/master_thesis/ggkm/utils/R-4.4.3}"
    if [[ -n "$R_HOME" ]]; then
        export LD_LIBRARY_PATH="$R_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    case "$job_key" in
        simulation-experiments) default_workers="${PBS_NCPUS:-4}" ;;
        melanoma-experiments) default_workers="${PBS_NCPUS:-4}" ;;
        breast-cancer-experiments) default_workers=2 ;;
    esac
    export PBS_NCPUS="${GGKM_WORKERS:-$default_workers}"
    if [[ ! "$PBS_NCPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "GGKM_WORKERS must be a positive integer; got $PBS_NCPUS." >&2
        return 1
    fi
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export PYTHONHASHSEED=0
    export MPLBACKEND=Agg

    printf 'Python: %s\nWorkers: %s\n' "$(command -v "$GGKM_PYTHON")" "$PBS_NCPUS"
    "$GGKM_PYTHON" --version
    "$GGKM_PYTHON" -u scripts/run_workflow.py --run-job "$job_key"
}
