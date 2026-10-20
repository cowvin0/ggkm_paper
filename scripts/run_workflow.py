#!/usr/bin/env python3
"""Run the experiment, figure, and table workflow locally or submit it to PBS.

Examples:
    python scripts/run_workflow.py --mode cluster --dry-run
    python scripts/run_workflow.py --mode cluster
    python scripts/run_workflow.py --mode local --stage postprocess
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workflow_utils import DATASETS, JOBS, PROJECT_ROOT, STAGES, Job, select_jobs


class WorkflowError(RuntimeError):
    """A failed command or invalid execution environment."""


def execution_environment(workers: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # A launcher can itself be invoked inside an interactive PBS allocation.
    # Array indices belong only to explicitly selected experiment tasks.
    for name in ("PBS_ARRAY_INDEX", "PBS_ARRAY_ID", "PBS_JOBID"):
        env.pop(name, None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    if workers is not None:
        env["PBS_NCPUS"] = str(workers)
        env["GGKM_WORKERS"] = str(workers)
    return env


def discover_task_counts(
    jobs: list[Job], python: str, env: dict[str, str]
) -> dict[str, int]:
    """Read current grids in the selected Python environment without running fits."""
    experiments = [job for job in jobs if job.stage == "experiments"]
    if not experiments:
        return {}
    modules = [job.commands[0][0] for job in experiments]
    code = (
        "import importlib,json,sys; "
        "print(json.dumps({name:len(importlib.import_module(name).build_experiments()) "
        "for name in sys.argv[1:]}))"
    )
    try:
        result = subprocess.run(
            [python, "-c", code, *modules], cwd=PROJECT_ROOT, env=env,
            check=True, text=True, capture_output=True,
        )
        counts = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(counts, dict):
            raise ValueError("Experiment grid discovery must return a JSON object")
        answer = {job.key: counts[job.commands[0][0]] for job in experiments}
        if any(type(count) is not int or count < 1 for count in answer.values()):
            raise ValueError("Each experiment grid must contain at least one task")
        return answer
    except (subprocess.CalledProcessError, IndexError, KeyError, ValueError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise WorkflowError(
            "Cannot read experiment grids. Activate the experiment environment or "
            f"select it with --python.\n{detail}"
        ) from error


def module_commands(job: Job, python: str) -> list[list[str]]:
    return [[python, "-u", "-m", *command] for command in job.commands]


def run_job(
    job: Job, python: str, env: dict[str, str], *, log_file: Path | None = None
) -> None:
    """Execute one array task or all commands of one postprocessing job."""
    if job.stage == "experiments":
        try:
            if int(env["PBS_ARRAY_INDEX"]) < 0:
                raise ValueError("negative index")
        except (KeyError, ValueError) as error:
            raise WorkflowError(f"{job.key} requires a nonnegative PBS_ARRAY_INDEX") from error

    def execute(stream=None):
        for command in module_commands(job, python):
            print(f"  $ {shlex.join(command)}", file=stream or sys.stdout, flush=True)
            try:
                subprocess.run(
                    command, cwd=PROJECT_ROOT, env=env, check=True,
                    stdout=stream, stderr=subprocess.STDOUT if stream else None,
                )
            except subprocess.CalledProcessError as error:
                where = f"; see {log_file}" if log_file else ""
                raise WorkflowError(
                    f"{job.key} failed (exit {error.returncode}){where}"
                ) from error

    if log_file is None:
        execute()
    else:
        with log_file.open("w", encoding="utf-8") as stream:
            execute(stream)


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_local(jobs, counts, python, env, log_dir, manifest):
    manifest_path = log_dir / "workflow.json"
    for job in jobs:
        total = counts.get(job.key, 1)
        record = {
            "job": job.key, "state": "running", "tasks": total,
            "completed_tasks": 0, "commands": module_commands(job, python),
        }
        manifest["jobs"].append(record)
        _write_manifest(manifest_path, manifest)
        try:
            for index in range(total):
                task_env = env.copy()
                suffix = ""
                if job.stage == "experiments":
                    task_env["PBS_ARRAY_INDEX"] = str(index)
                    suffix = f"_{index:04d}"
                logfile = log_dir / f"{job.key}{suffix}.log"
                print(f"Running {job.key}{suffix} ({index + 1}/{total}); log: {logfile}", flush=True)
                run_job(job, python, task_env, log_file=logfile)
                record["completed_tasks"] += 1
                _write_manifest(manifest_path, manifest)
        except (OSError, WorkflowError, KeyboardInterrupt) as error:
            record.update(state="failed", error=str(error))
            _write_manifest(manifest_path, manifest)
            raise
        record["state"] = "complete"
        _write_manifest(manifest_path, manifest)


def qsub_command(job, counts, submitted, qsub, queue, export_names):
    command = [qsub]
    if queue:
        command.extend(["-q", queue])
    command.extend(["-v", ",".join(export_names)])
    if job.stage == "experiments":
        command.extend(["-J", f"0-{counts[job.key] - 1}"])
    if job.prerequisite in submitted:
        command.extend(["-W", f"depend=afterok:{submitted[job.prerequisite]}"])
    command.append(str(PROJECT_ROOT / "pbs_jobs" / job.pbs_file))
    return command


def submit_cluster(jobs, counts, qsub, queue, env, export_names, log_dir, manifest):
    submitted = {}
    manifest_path = log_dir / "workflow.json"
    for job in jobs:
        command = qsub_command(job, counts, submitted, qsub, queue, export_names)
        record = {"job": job.key, "command": command, "state": "submitting"}
        manifest["jobs"].append(record)
        _write_manifest(manifest_path, manifest)
        try:
            result = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True, check=True,
            )
            job_id = result.stdout.strip()
            # Preserve the [] in array IDs and the server suffix for dependencies.
            if not re.fullmatch(r"\d+(?:\[\])?(?:\.[A-Za-z0-9_.-]+)?", job_id):
                raise WorkflowError(f"qsub returned an unrecognized job ID: {job_id!r}")
        except (OSError, subprocess.CalledProcessError, WorkflowError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            record.update(state="failed", error=detail)
            _write_manifest(manifest_path, manifest)
            raise WorkflowError(
                f"Could not submit {job.key}: {detail}\n"
                f"Earlier submissions remain recorded in {manifest_path}"
            ) from error
        submitted[job.key] = job_id
        record.update(state="submitted", job_id=job_id)
        _write_manifest(manifest_path, manifest)
        print(f"Submitted {job.key}: {job_id}", flush=True)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("local", "cluster"), help="Run locally or submit to PBS (required).")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--stage", choices=STAGES, default="all", help="Use postprocess for existing experiment results.")
    parser.add_argument("--python", default=sys.executable, help="Python executable with the experiment dependencies installed.")
    parser.add_argument("--workers", type=positive_int, help="Outer-holdout workers per task (local default: 1; cluster: PBS job defaults, at most 4).")
    parser.add_argument("--queue", help="PBS queue (cluster only; otherwise use the scheduler default).")
    parser.add_argument("--qsub", default="qsub", help="PBS qsub executable (cluster only).")
    parser.add_argument("--log-dir", type=Path, help="New directory for this run's logs and workflow.json.")
    parser.add_argument("--dry-run", action="store_true", help="Show commands and current array sizes without running or submitting jobs.")
    parser.add_argument("--run-job", choices=tuple(JOBS), help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_job and args.mode is None:
        parser.error("--mode is required; choose local or cluster")
    if args.mode == "local" and args.queue:
        parser.error("--queue requires --mode cluster")
    if args.mode == "cluster" and args.workers and args.workers > 4:
        parser.error("the PBS experiment jobs request 4 CPUs; --workers must be at most 4")
    try:
        python = shutil.which(args.python)
        if python is None:
            raise WorkflowError(f"Python executable not found: {args.python}")
        python = os.path.abspath(python)  # Retain a virtual environment's symlink.
        if args.run_job:
            # Workers retain the scheduler's index and allocated/configured CPUs.
            job = JOBS[args.run_job]
            if args.dry_run:
                for command in module_commands(job, python):
                    print(shlex.join(command))
                return 0
            env = execution_environment(args.workers)
            if job.stage == "experiments" and "PBS_ARRAY_INDEX" in os.environ:
                env["PBS_ARRAY_INDEX"] = os.environ["PBS_ARRAY_INDEX"]
            run_job(job, python, env)
            return 0

        jobs = select_jobs(args.datasets, args.stage)
        workers = (args.workers or 1) if args.mode == "local" else args.workers
        if args.mode == "cluster" and workers is None and os.environ.get("GGKM_WORKERS"):
            try:
                workers = int(os.environ["GGKM_WORKERS"])
            except ValueError as error:
                raise WorkflowError("GGKM_WORKERS must be a positive integer, at most 4") from error
            if not 1 <= workers <= 4:
                raise WorkflowError("GGKM_WORKERS must be a positive integer, at most 4")
        env = execution_environment(workers)
        counts = discover_task_counts(jobs, python, env)
        env.update(GGKM_PROJECT_ROOT=str(PROJECT_ROOT), GGKM_PYTHON=python)
        export_names = sorted(name for name in env if name.startswith("GGKM_"))
        print(f"{args.mode.capitalize()} workflow: {', '.join(args.datasets)}; stage={args.stage}", flush=True)
        for job in jobs:
            count = counts.get(job.key)
            print(f"  {job.key}" + (f": {count} tasks (indices 0-{count - 1})" if count else ""), flush=True)

        if args.dry_run:
            submitted = {}
            for job in jobs:
                if args.mode == "cluster":
                    command = qsub_command(job, counts, submitted, args.qsub, args.queue, export_names)
                    print(shlex.join(command))
                    submitted[job.key] = f"<{job.key}-job-id>"
                else:
                    if job.stage == "experiments":
                        print(f"# Repeat with PBS_ARRAY_INDEX=0 through {counts[job.key] - 1}")
                    for command in module_commands(job, python):
                        print(shlex.join(command))
            return 0

        qsub = None
        if args.mode == "cluster":
            qsub = shutil.which(args.qsub)
            if qsub is None:
                raise WorkflowError(f"PBS submission executable not found: {args.qsub}")
            for job in jobs:
                if not (PROJECT_ROOT / "pbs_jobs" / job.pbs_file).is_file():
                    raise WorkflowError(f"Missing PBS job: {job.pbs_file}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        log_dir = args.log_dir or PROJECT_ROOT / "logs" / "workflow" / f"{stamp}_{os.getpid()}"
        log_dir = log_dir.absolute()
        log_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "mode": args.mode, "datasets": args.datasets, "stage": args.stage,
            "python": python, "task_counts": counts, "jobs": [],
        }
        _write_manifest(log_dir / "workflow.json", manifest)
        if args.mode == "local":
            run_local(jobs, counts, python, env, log_dir, manifest)
        else:
            submit_cluster(jobs, counts, qsub, args.queue, env, export_names, log_dir, manifest)
        print(f"Workflow record: {log_dir / 'workflow.json'}")
        return 0
    except (OSError, WorkflowError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Workflow interrupted; completed tasks/submissions remain in workflow.json.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
