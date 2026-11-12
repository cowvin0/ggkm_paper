#!/usr/bin/env python3
"""Run the separate confirmatory robustness PBS/local dependency chain."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import sys

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import confirmatory_robustness_utils as confirm
from scripts.run_workflow import execution_environment, qsub_command, run_local, submit_cluster, WorkflowError


PROJECT_ROOT = confirm.original.PROJECT_ROOT


@dataclass(frozen=True)
class ConfirmatoryJob:
    stage: str
    prerequisite: str | None
    count: int | None = None
    dataset: str = "latent-confirmatory"

    @property
    def key(self):
        return f"confirmatory-{self.stage}"

    @property
    def is_array(self):
        return self.count is not None

    @property
    def commands(self):
        return (("scripts.confirmatory_robustness", "--stage", self.stage),)

    @property
    def pbs_file(self):
        return f"confirmatory_{self.stage.replace('-', '_')}.pbs"


def jobs_for(stage):
    jobs = [ConfirmatoryJob("audit", None, len(confirm.AUDIT_CASE_IDS)),
            ConfirmatoryJob("audit-summary", "confirmatory-audit"),
            ConfirmatoryJob("pilot", "confirmatory-audit-summary", 48),
            ConfirmatoryJob("pilot-summary", "confirmatory-pilot"),
            ConfirmatoryJob("final", "confirmatory-pilot-summary", 48),
            ConfirmatoryJob("report", "confirmatory-final")]
    return {"all": jobs, "audit": jobs[:2], "pilot": jobs[:4], "final": jobs[4:], "report": jobs[5:]}[stage]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "cluster"), required=True)
    parser.add_argument("--stage", choices=("audit", "pilot", "final", "all", "report"), default="audit")
    parser.add_argument("--protocol", type=Path, default=confirm.DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--queue")
    parser.add_argument("--qsub", default="qsub")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = confirm.load_protocol(args.protocol)
        if args.stage != "audit":
            confirm.require_margins(config)
        workers = args.workers
        if args.mode == "cluster" and workers is None and os.environ.get("GGKM_WORKERS"):
            workers = int(os.environ["GGKM_WORKERS"])
        if workers is not None and (workers < 1 or args.mode == "cluster" and workers > 4):
            raise ValueError("Workers must be positive and at most 4 for the PBS jobs")
        if args.queue and args.mode != "cluster":
            raise ValueError("--queue requires --mode cluster")
        python = shutil.which(args.python)
        if python is None:
            raise ValueError(f"Python interpreter not found: {args.python}")
        python = os.path.abspath(python)
        env = execution_environment((workers or 1) if args.mode == "local" else workers)
        env.update(GGKM_PROJECT_ROOT=str(PROJECT_ROOT), GGKM_PYTHON=python,
                   GGKM_CONFIRMATORY_PROTOCOL=str(args.protocol.resolve()),
                   GGKM_CONFIRMATORY_DIR=str((args.output_dir or confirm.results_root()).resolve()))
        jobs = jobs_for(args.stage)
        counts = {job.key: job.count for job in jobs if job.is_array}
        export_names = sorted(name for name in env if name.startswith("GGKM_"))
        if args.dry_run:
            submitted = {}
            for job in jobs:
                if args.mode == "cluster":
                    print(shlex.join(qsub_command(job, counts, submitted, args.qsub, args.queue, export_names)))
                    submitted[job.key] = f"<{job.key}-job-id>"
                else:
                    print(f"{job.key}: {job.count or 1} task(s): " + shlex.join([python, "-m", *job.commands[0]]))
            return 0
        log_dir = args.log_dir or PROJECT_ROOT / "logs" / f"confirmatory_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}"
        log_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"study": "latent-confirmatory", "stage": args.stage, "mode": args.mode,
                    "protocol": env["GGKM_CONFIRMATORY_PROTOCOL"], "output_dir": env["GGKM_CONFIRMATORY_DIR"],
                    "task_counts": counts, "jobs": []}
        if args.mode == "cluster":
            submit_cluster(jobs, counts, args.qsub, args.queue, env, export_names, log_dir, manifest)
        else:
            run_local(jobs, counts, python, env, log_dir, manifest)
        return 0
    except (OSError, ValueError, KeyError, WorkflowError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
