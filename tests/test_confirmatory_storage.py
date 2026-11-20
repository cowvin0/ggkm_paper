"""Compact storage preserves statistical outputs, evidence checks, and gates."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import confirmatory_robustness as runner
from scripts import confirmatory_robustness_storage as storage
from scripts import confirmatory_robustness_utils as confirm
from scripts import summarize_latent_robustness as exploratory
from tests.test_confirmatory_robustness import config, completed_study


def test_trace_compaction_keeps_failures_and_objectives_without_iteration_growth(tmp_path):
    iterations = [dict(stage="latency", success=False, status=2, message="numerical failure"),
                  dict(stage="incidence", success=True, status=0, message="converged")]
    payload = dict(status="ok", converged=False, final_loglik=-8.,
                   fits=[dict(attempts=[dict(final_loglik=-8., loglik_history=[-10., -9., -9.1, -8.],
                                            optimizer_results=iterations * 5000)])])
    compact = storage.summarize_traces(payload)
    attempt = compact["fits"][0]["attempts"][0]
    assert compact["converged"] is False and compact["final_loglik"] == -8.
    assert attempt["loglik_history_summary"] == dict(count=4, finite=4, first=-10., last=-8.,
                                                    minimum=-10., maximum=-8., decreases=1)
    assert attempt["optimizer_summary"]["latency"]["failed"] == 5000
    assert attempt["optimizer_summary"]["incidence"]["failed"] == 0
    assert len(json.dumps(compact)) < len(json.dumps(payload)) / 100
    assert "optimizer_results" in payload["fits"][0]["attempts"][0]  # Input is untouched.
    audit = dict(stage="audit", confirmatory_protocol={"storage": {"format": "compact-v1"}})
    assert storage.diagnostic_payload(payload, audit) == payload  # Full audit traces are retained.
    payload["fits"] *= 2
    payload["fits"][0]["selected_attempt"] = 0
    pilot = {**audit, "stage": "pilot"}
    retained = storage.diagnostic_payload(payload, pilot)
    assert retained["fits"][0]["attempt_summary"]["attempted"] == 1
    assert retained["fits"][0]["selected_start"]["final_loglik"] == -8.
    assert "attempts" not in retained["fits"][0]
    assert retained["fits"][-1]["attempts"][0]["final_loglik"] == -8.


def test_compressed_json_is_deterministic_and_failed_writes_preserve_previous_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.json.gz"
    storage.write_json(path, {"valid": True, "rows": [1, 2, 3]})
    before = path.read_bytes()
    storage.write_json(path, {"valid": True, "rows": [1, 2, 3]})
    assert path.read_bytes() == before
    with pytest.raises(OSError, match="No space"):
        with storage.atomic_text(path) as handle:
            handle.write("partial output")
            raise OSError(28, "No space left on device")
    assert path.read_bytes() == before
    assert storage.read_json(path)["rows"] == [1, 2, 3]
    assert not list(tmp_path.glob("*.tmp"))


def test_archive_disk_failure_is_not_cached_as_a_scientific_failure(tmp_path, monkeypatch):
    design = dict(stage="audit", study_id="test", confirmatory_protocol={"storage": {"format": "compact-v1"}})
    monkeypatch.setattr(runner, "prepare_replication", lambda *args: ({}, {}, {"array": [1, 2]}))

    def no_space(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner, "atomic_write_npz", no_space)
    with pytest.raises(OSError, match="No space"):
        runner.run_replication({"task_id": 0}, 0, design, tmp_path, {})
    assert not list(tmp_path.rglob("replication_*.json*"))


@pytest.mark.parametrize("completed_study", ["compact-v1"], indirect=True)
def test_compact_pipeline_matches_in_memory_statistics_and_validates_samples(completed_study):
    root, config = completed_study
    design, frame = runner.read_results("final", root, config)
    pairs = confirm.paired_values(frame)
    expected = confirm.contrast_summary(frame, pairs, config)
    performance = exploratory.performance_summary(frame)
    report = runner.summarize("final", root, config, figures=False)
    assert report["decision"] == "within_prespecified_margins"
    assert report["attempted"] == len(frame)
    assert report["converged"] == int(frame.converged.sum())
    actual = pd.read_csv(root / "final/paired_summary.csv")
    order = ["task_id", "latency_mode", "model_name", "analysis", "metric"]
    pd.testing.assert_frame_equal(actual.sort_values(order).reset_index(drop=True),
                                  expected.sort_values(order).reset_index(drop=True), check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    saved_pairs = pd.read_csv(root / "final/paired_replications.csv.gz")
    order += ["replication"]
    pd.testing.assert_frame_equal(saved_pairs.sort_values(order).reset_index(drop=True),
                                  pairs.sort_values(order).reset_index(drop=True), check_dtype=False)
    actual = pd.read_csv(root / "final/performance_summary.csv")
    expected_performance = pd.read_csv(io.StringIO(performance.to_csv(index=False)))
    pd.testing.assert_frame_equal(actual, expected_performance, check_dtype=False)
    assert not (root / "final/samples").exists()
    assert not (root / "pilot/samples").exists()
    assert list((root / "audit/samples").rglob("*.npz"))
    allocation = storage.read_json(root / "pilot/allocation.json")
    assert allocation["precision_ready"] and allocation["storage_ready"]
    repeated = runner.summarize("pilot", root, config, figures=False)
    assert repeated["report_id"] == allocation["report_id"]
    plan = storage.read_json(root / "pilot/storage_plan.json")
    assert plan["within_budget"] and plan["projected_gib"] < 80

    # Cached successful replications are reused without fitting again.
    case = design["cases"][0]
    path = root / "final/checkpoints" / f"case_{case['task_id']:03d}/replication_0000.json.gz"
    original = path.read_bytes()
    saved = storage.read_json(path)
    runner.run_replication(case, 0, design, root / "final", {})
    assert path.read_bytes() == original
    saved["sample_array_hash"] = "tampered"
    storage.write_json(path, saved)
    with pytest.raises(ValueError, match="Regenerated sample differs"):
        runner.run_replication(case, 0, design, root / "final", {})
    path.write_bytes(original)
    saved = storage.read_json(path)
    saved["rows"][0]["data_hash"] = "tampered"
    storage.write_json(path, saved)
    with pytest.raises(ValueError, match="Regenerated pairing differs"):
        runner.run_replication(case, 0, design, root / "final", {})


@pytest.mark.parametrize("completed_study", ["compact-v1"], indirect=True)
def test_storage_budget_blocks_final_without_changing_replication_allocation(completed_study, monkeypatch):
    root, config = completed_study
    old = storage.read_json(root / "pilot/allocation.json")
    monkeypatch.setattr(storage, "storage_projection", lambda *args: dict(within_budget=False, projected_gib=120., budget_gib=80.))
    report = runner.summarize("pilot", root, config, figures=False)
    assert report["precision_ready"]
    assert not report["storage_ready"] and not report["ready"]
    assert report["final_replications"] == old["final_replications"]
    with pytest.raises(ValueError, match="Prerequisite is not ready"):
        runner.build_design("final", root, config)
