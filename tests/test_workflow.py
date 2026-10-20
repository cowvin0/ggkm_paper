"""Exercise workflow orchestration without fitting models or submitting jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "run_workflow.py"


def _launch(*arguments, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, str(LAUNCHER), *arguments],
        cwd=cwd or PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def fake_commands(tmp_path):
    """Supply disposable Python/qsub stand-ins that record real CLI invocations."""
    calls = tmp_path / "commands.jsonl"
    python = tmp_path / "fake-python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "capture = pathlib.Path(os.environ['WORKFLOW_TEST_CALLS'])\n"
        "if args[0] == '-c':\n"
        "    event = {'kind': 'discovery', 'modules': args[2:]}\n"
        "    with capture.open('a') as stream: stream.write(json.dumps(event) + '\\n')\n"
        "    count = json.loads(os.environ.get('WORKFLOW_TEST_TASK_COUNT', '2'))\n"
        "    payload = os.environ.get('WORKFLOW_TEST_DISCOVERY_PAYLOAD')\n"
        "    print(payload if payload is not None else json.dumps({module: count for module in args[2:]}))\n"
        "    raise SystemExit(0)\n"
        "event = {'kind': 'module', 'argv': args, 'cwd': os.getcwd(),\n"
        "    'index': os.environ.get('PBS_ARRAY_INDEX'), 'job_id': os.environ.get('PBS_JOBID'),\n"
        "    'workers': os.environ.get('PBS_NCPUS')}\n"
        "with capture.open('a') as stream: stream.write(json.dumps(event) + '\\n')\n"
        "if os.environ.get('WORKFLOW_TEST_FAIL_MODULE') in args:\n"
        "    print('synthetic fit failure', file=sys.stderr)\n"
        "    raise SystemExit(7)\n"
    )
    python.chmod(0o755)
    qsub_calls = tmp_path / "qsub.jsonl"
    qsub = tmp_path / "fake-qsub"
    qsub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "capture = pathlib.Path(os.environ['WORKFLOW_TEST_QSUB'])\n"
        "previous = capture.read_text().splitlines() if capture.exists() else []\n"
        "args = sys.argv[1:]\n"
        "number = 101 + len(previous)\n"
        "job_id = str(number) + ('[].cluster' if '-J' in args else '.cluster')\n"
        "event = {'argv': args, 'job_id': job_id, 'cwd': os.getcwd(),\n"
        "    'project_root': os.environ.get('GGKM_PROJECT_ROOT'),\n"
        "    'workers': os.environ.get('GGKM_WORKERS')}\n"
        "with capture.open('a') as stream: stream.write(json.dumps(event) + '\\n')\n"
        "if os.environ.get('WORKFLOW_TEST_QSUB_FAIL_AT') == str(number):\n"
        "    print('synthetic scheduler rejection', file=sys.stderr)\n"
        "    raise SystemExit(9)\n"
        "print(job_id)\n"
    )
    qsub.chmod(0o755)
    environment = dict(
        os.environ,
        WORKFLOW_TEST_CALLS=str(calls),
        WORKFLOW_TEST_QSUB=str(qsub_calls),
        PBS_ARRAY_INDEX="stale-index",
        PBS_JOBID="stale-job.cluster",
        PBS_NCPUS="99",
    )
    return {
        "calls": calls, "qsub_calls": qsub_calls,
        "python": python, "qsub": qsub,
        "environment": environment, "logs": tmp_path / "workflow-logs",
    }


def _with_fake_python(fake, *arguments, cwd=None):
    return _launch(
        *arguments, "--python", str(fake["python"]),
        "--log-dir", str(fake["logs"]),
        cwd=cwd, env=fake["environment"],
    )


def _modules(fake):
    return [event for event in _events(fake["calls"]) if event["kind"] == "module"]


def test_launcher_help_works_from_outside_project(tmp_path):
    process = _launch("--help", cwd=tmp_path)

    assert process.returncode == 0, process.stderr
    for option in ("--mode", "--datasets", "--stage", "--python", "--workers", "--dry-run"):
        assert option in process.stdout


def test_launcher_requires_explicit_execution_mode():
    process = _launch()

    assert process.returncode != 0
    assert "--mode" in process.stderr


@pytest.mark.parametrize("workers", ["0", "-1", "1.5"])
def test_invalid_worker_counts_fail_before_work(workers):
    process = _launch("--mode", "local", "--stage", "postprocess", "--workers", workers)

    assert process.returncode != 0
    assert "--workers" in process.stderr


def test_cluster_postprocess_dry_run_needs_no_scheduler_and_creates_no_files(tmp_path):
    capture = tmp_path / "qsub.jsonl"
    qsub = tmp_path / "qsub"
    qsub.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib\n"
        "pathlib.Path(os.environ['WORKFLOW_TEST_QSUB']).write_text('called')\n"
        "raise SystemExit(73)\n"
    )
    qsub.chmod(0o755)
    environment = dict(os.environ, PATH=str(tmp_path), WORKFLOW_TEST_QSUB=str(capture))
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    process = _launch("--mode", "cluster", "--stage", "postprocess", "--dry-run", cwd=tmp_path, env=environment)

    assert process.returncode == 0, process.stderr
    assert "qsub" in process.stdout
    for dataset in ("simulation", "melanoma", "breast-cancer"):
        assert f"{dataset}-figures" in process.stdout
        assert f"{dataset}-tables" in process.stdout
    assert not capture.exists()
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_local_complete_dataset_runs_each_task_before_figures_and_tables(fake_commands, tmp_path):
    fake = fake_commands
    process = _with_fake_python(fake, "--mode", "local", "--datasets", "simulation", cwd=tmp_path)

    assert process.returncode == 0, process.stderr
    events = _modules(fake)
    assert [event["argv"][2] for event in events] == [
        "scripts.run_pcm_experiment", "scripts.run_pcm_experiment",
        "scripts.create_paired_plot_1_2_3_methods",
        "scripts.create_paired_plot_interaction_quadratic_surfaces",
        "scripts.recover_cure_surfaces_figure",
        "scripts.create_simulation_tables",
    ]
    assert [event["index"] for event in events] == ["0", "1", None, None, None, None]
    assert all(event["job_id"] is None for event in events)
    assert all(event["cwd"] == str(PROJECT_ROOT) for event in events)
    assert all(event["workers"] == "1" for event in events)
    assert list(fake["logs"].rglob("workflow.json"))


@pytest.mark.parametrize("dataset", ["melanoma", "breast-cancer"])
def test_local_clinical_postprocess_uses_matching_figures_and_table(fake_commands, dataset):
    fake = fake_commands
    process = _with_fake_python(
        fake, "--mode", "local", "--datasets", dataset,
        "--stage", "postprocess", "--workers", "3",
    )

    assert process.returncode == 0, process.stderr
    events = _modules(fake)
    prefix = dataset.replace("-", "_")
    assert [event["argv"][2] for event in events] == [
        f"scripts.{prefix}_auc_plot", f"scripts.{prefix}_individual_cure_variation",
        "scripts.create_clinical_performance_tables",
    ]
    assert events[-1]["argv"][3:] == ["--dataset", dataset]
    assert all(event["index"] is None and event["job_id"] is None for event in events)
    assert all(event["workers"] == "3" for event in events)
    assert not any(event["kind"] == "discovery" for event in _events(fake["calls"]))


def test_local_failure_prevents_remaining_tasks_and_postprocessing(fake_commands):
    fake = fake_commands
    fake["environment"]["WORKFLOW_TEST_FAIL_MODULE"] = "scripts.run_melanoma_experiment"

    process = _with_fake_python(fake, "--mode", "local", "--datasets", "melanoma")

    assert process.returncode != 0
    events = _modules(fake)
    assert len(events) == 1
    assert events[0]["index"] == "0"
    assert "synthetic fit failure" in "".join(path.read_text() for path in fake["logs"].rglob("*.log"))
    manifest = json.loads((fake["logs"] / "workflow.json").read_text())
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"][0]["state"] == "failed"
    assert manifest["jobs"][0]["completed_tasks"] == 0


def test_local_dry_run_discovers_current_grid_without_creating_logs(fake_commands):
    fake = fake_commands

    process = _with_fake_python(fake, "--mode", "local", "--datasets", "melanoma", "--dry-run")

    assert process.returncode == 0, process.stderr
    events = _events(fake["calls"])
    assert len(events) == 1
    assert events[0] == {"kind": "discovery", "modules": ["scripts.run_melanoma_experiment"]}
    assert not fake["logs"].exists()
    assert "melanoma-experiments" in process.stdout
    assert "melanoma-figures" in process.stdout
    assert "melanoma-tables" in process.stdout


def test_cluster_submits_dynamic_arrays_and_matching_dependents(fake_commands):
    fake = fake_commands

    process = _with_fake_python(
        fake, "--mode", "cluster", "--qsub", str(fake["qsub"]), "--queue", "research", "--workers", "2",
    )

    assert process.returncode == 0, process.stderr
    submissions = _events(fake["qsub_calls"])
    assert len(submissions) == 9
    arrays = [event for event in submissions if "-J" in event["argv"]]
    outputs = [event for event in submissions if "-J" not in event["argv"]]
    assert len(arrays) == 3
    assert len(outputs) == 6
    for event in arrays:
        arguments = event["argv"]
        assert arguments[arguments.index("-J") + 1] == "0-1"
        assert not any("depend=" in argument for argument in arguments)
    for event in submissions:
        arguments = event["argv"]
        assert arguments[arguments.index("-q") + 1] == "research"
        assert event["cwd"] == str(PROJECT_ROOT)
        assert event["project_root"] == str(PROJECT_ROOT)
        assert event["workers"] == "2"
    for parent in arrays:
        assert sum(parent["job_id"] in argument for event in outputs for argument in event["argv"]) == 2
    assert all(any("depend=afterok:" in argument for argument in event["argv"]) for event in outputs)
    by_script = {Path(event["argv"][-1]).name: event for event in submissions}
    for experiment, figures, tables in [
        ("run_experiment.pbs", "create_simulation_figures.pbs", "create_simulation_tables.pbs"),
        ("run_melanoma_experiment.pbs", "create_melanoma_figures.pbs", "create_melanoma_table.pbs"),
        ("run_breast_cancer_experiment.pbs", "create_breast_cancer_figures.pbs", "create_breast_cancer_table.pbs"),
    ]:
        parent_id = by_script[experiment]["job_id"]
        assert f"depend=afterok:{parent_id}" in by_script[figures]["argv"]
        assert f"depend=afterok:{parent_id}" in by_script[tables]["argv"]
    assert _modules(fake) == []


def test_cluster_postprocess_submits_without_array_dependencies(fake_commands):
    fake = fake_commands

    process = _with_fake_python(
        fake, "--mode", "cluster", "--qsub", str(fake["qsub"]),
        "--datasets", "breast-cancer", "--stage", "postprocess",
    )

    assert process.returncode == 0, process.stderr
    submissions = _events(fake["qsub_calls"])
    assert len(submissions) == 2
    assert all("-J" not in event["argv"] for event in submissions)
    assert all(not any("depend=" in argument for argument in event["argv"]) for event in submissions)
    assert _events(fake["calls"]) == []


def test_cluster_failure_preserves_prior_job_ids_and_stops_submitting(fake_commands):
    fake = fake_commands
    fake["environment"]["WORKFLOW_TEST_QSUB_FAIL_AT"] = "102"

    process = _with_fake_python(fake, "--mode", "cluster", "--qsub", str(fake["qsub"]))

    assert process.returncode != 0
    assert "synthetic scheduler rejection" in process.stderr
    submissions = _events(fake["qsub_calls"])
    assert len(submissions) == 2
    assert all("-J" in event["argv"] for event in submissions)
    manifest = json.loads((fake["logs"] / "workflow.json").read_text())
    assert [job["state"] for job in manifest["jobs"]] == ["submitted", "failed"]
    assert [job["job_id"] for job in manifest["jobs"][:-1]] == [event["job_id"] for event in submissions[:-1]]


@pytest.mark.parametrize("count", ["0", "-1", "1.5", '"2"', "true"])
def test_invalid_discovered_grids_fail_without_submission_or_logs(fake_commands, count):
    fake = fake_commands
    fake["environment"]["WORKFLOW_TEST_TASK_COUNT"] = count

    process = _with_fake_python(fake, "--mode", "cluster", "--qsub", str(fake["qsub"]))

    assert process.returncode != 0
    assert "experiment grids" in process.stderr
    assert not fake["qsub_calls"].exists()
    assert not fake["logs"].exists()
    assert _modules(fake) == []


def test_pbs_worker_uses_only_the_scheduler_selected_experiment_task(fake_commands):
    fake = fake_commands
    fake["environment"]["PBS_ARRAY_INDEX"] = "7"

    process = _with_fake_python(fake, "--run-job", "simulation-experiments")

    assert process.returncode == 0, process.stderr
    events = _modules(fake)
    assert len(events) == 1
    assert events[0]["argv"][2] == "scripts.run_pcm_experiment"
    assert events[0]["index"] == "7"
    assert _events(fake["calls"]) == events
    assert not fake["logs"].exists()


def test_pbs_postprocess_worker_clears_inherited_array_index(fake_commands):
    fake = fake_commands
    fake["environment"]["PBS_ARRAY_INDEX"] = "7"

    process = _with_fake_python(fake, "--run-job", "melanoma-figures")

    assert process.returncode == 0, process.stderr
    events = _modules(fake)
    assert len(events) == 2
    assert all(event["index"] is None and event["job_id"] is None for event in events)
    assert not fake["logs"].exists()


def test_pbs_experiment_worker_requires_an_array_index(fake_commands):
    fake = fake_commands
    fake["environment"].pop("PBS_ARRAY_INDEX")

    process = _with_fake_python(fake, "--run-job", "melanoma-experiments")

    assert process.returncode != 0
    assert "PBS_ARRAY_INDEX" in process.stderr
    assert _events(fake["calls"]) == []


def test_pbs_worker_dry_run_does_not_execute_a_module(fake_commands):
    fake = fake_commands

    process = _with_fake_python(fake, "--run-job", "simulation-experiments", "--dry-run")

    assert process.returncode == 0, process.stderr
    assert "scripts.run_pcm_experiment" in process.stdout
    assert _events(fake["calls"]) == []
    assert not fake["logs"].exists()


@pytest.mark.parametrize("payload", ["null", "[]", '"unexpected"'])
def test_nonobject_discovery_output_has_an_actionable_error(fake_commands, payload):
    fake = fake_commands
    fake["environment"]["WORKFLOW_TEST_DISCOVERY_PAYLOAD"] = payload

    process = _with_fake_python(fake, "--mode", "cluster", "--qsub", str(fake["qsub"]))

    assert process.returncode != 0
    assert "experiment grids" in process.stderr
    assert "Traceback" not in process.stderr
    assert not fake["qsub_calls"].exists()
    assert not fake["logs"].exists()


@pytest.mark.parametrize("workers", ["0", "5", "invalid"])
def test_invalid_inherited_cluster_workers_fail_before_discovery_or_submission(fake_commands, workers):
    fake = fake_commands
    fake["environment"]["GGKM_WORKERS"] = workers

    process = _with_fake_python(fake, "--mode", "cluster", "--qsub", str(fake["qsub"]))

    assert process.returncode != 0
    assert "GGKM_WORKERS" in process.stderr
    assert _events(fake["calls"]) == []
    assert not fake["qsub_calls"].exists()
    assert not fake["logs"].exists()


@pytest.mark.parametrize("stage, modules", [
    ("experiments", ["scripts.run_melanoma_experiment"] * 2),
    ("figures", ["scripts.melanoma_auc_plot", "scripts.melanoma_individual_cure_variation"]),
    ("tables", ["scripts.create_clinical_performance_tables"]),
])
def test_local_stage_selection_runs_only_requested_work(fake_commands, stage, modules):
    fake = fake_commands

    process = _with_fake_python(fake, "--mode", "local", "--datasets", "melanoma", "--stage", stage)

    assert process.returncode == 0, process.stderr
    assert [event["argv"][2] for event in _modules(fake)] == modules
