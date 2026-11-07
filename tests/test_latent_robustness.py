"""Validate count-law generation, pairing, resumability, and robustness reports."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

from scripts import latent_robustness_utils as study
from scripts import run_latent_robustness_experiment as runner
from scripts import summarize_latent_robustness as reporting
from scripts.workflow_utils import select_jobs


def test_design_has_40_primary_8_stress_and_paired_fixed_latency_diagnostics():
    cases = study.build_experiments()
    assert len(cases) == 48
    assert sum(case["study_part"] == "primary" for case in cases) == 40
    assert sum(case["study_part"] == "stress" for case in cases) == 8
    assert sum("fixed" in study.latency_modes(case) for case in cases) == 20
    assert sum(len(runner.expected_fits(case)) for case in cases) == 204
    design = study.build_design()
    assert design["replications"] == 250
    assert design["parameter_grids"] == {model: study.pcm.parameter_grid(model, "rbf") for model in study.FIT_MODELS}
    for case in cases:
        assert case["method"] in {2, 3}
        assert case["n"] in {300, 600}


@pytest.mark.parametrize("truth", [truth["name"] for truth in study.TRUTHS])
def test_count_laws_preserve_cure_surface_and_generate_their_population_survival(truth):
    case = {**study.build_experiments()[0], "truth": truth, "n": 6000}
    design = study.build_design(replications=2)
    data, _ = study.generate_replication(case, 0, design, 3.5)
    counts = data["true_count"]
    assert np.issubdtype(counts.dtype, np.integer)
    assert np.all(counts >= 0)
    assert np.array_equal(np.isinf(data["true_time"]), counts == 0)
    if truth == "binomial2":
        assert counts.max() <= 2
    times = np.array([0.0, 0.5, 1.5, 3.0, np.inf])
    survival = study.true_survival(data["X"], times, case)
    np.testing.assert_allclose(survival[:, 0], 1)
    np.testing.assert_allclose(survival[:, -1], data["true_cure"], rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(survival, axis=1) <= 1e-12)
    for index, time in enumerate(times[1:-1], start=1):
        assert np.mean(data["true_time"] > time) == pytest.approx(np.mean(survival[:, index]), abs=0.035)
    assert np.mean(counts == 0) == pytest.approx(np.mean(data["true_cure"]), abs=0.035)
    np.testing.assert_array_equal(data["event"], data["true_time"] <= data["censor_time"])


def test_followup_calibration_targets_susceptible_censoring_with_independent_samples():
    long, short = study.build_experiments()[:2]
    first = study.calibrate_censoring(long, sample_size=8192)
    second = study.calibrate_censoring(short, sample_size=8192)
    assert first["censor_mean"] > second["censor_mean"]
    assert first["calibration_censoring_fraction"] == pytest.approx(0.2, abs=1e-10)
    assert second["calibration_censoring_fraction"] == pytest.approx(0.5, abs=1e-10)
    case = {**short, "n": 6000}
    data, _ = study.generate_replication(case, 3, study.build_design(replications=2), second["censor_mean"])
    susceptible = data["true_count"] > 0
    assert np.mean(data["event"][susceptible] == 0) == pytest.approx(0.5, abs=0.04)
    assert np.mean(data["event"] == 0) >= np.mean(~susceptible)


def test_pilot_seeds_are_distinct_and_model_law_never_changes_data_or_splits():
    case = study.build_experiments()[0]
    final = study.replication_seeds(case, 0, study.MASTER_SEED)
    pilot = study.replication_seeds(case, 0, study.PILOT_MASTER_SEED)
    assert all(final[name] != pilot[name] for name in study.SEED_COLUMNS)
    independent_stream = study.replication_seeds(case, 0, study.MASTER_SEED, phase="pilot")
    assert all(final[name] != independent_stream[name] for name in study.SEED_COLUMNS)
    changed = study.replication_seeds({**case, "fitted_model": "binomial"}, 0, study.MASTER_SEED)
    assert changed == final
    long, short = study.build_experiments()[:2]
    assert study.replication_seeds(long, 0, study.MASTER_SEED) == study.replication_seeds(short, 0, study.MASTER_SEED)


def test_final_allocation_can_change_without_invalidating_independent_pilot(monkeypatch, tmp_path):
    monkeypatch.delenv("GGKM_ROBUSTNESS_REPLICATIONS", raising=False)
    pilot = study.build_design("pilot")
    monkeypatch.setenv("GGKM_ROBUSTNESS_REPLICATIONS", "500")
    assert study.build_design("pilot")["study_id"] == pilot["study_id"]
    assert study.build_design("final")["replications"] == 500
    monkeypatch.setenv("GGKM_ROBUSTNESS_FINAL_DIR", str(tmp_path / "final-500"))
    assert study.phase_directory("final") == tmp_path / "final-500"
    monkeypatch.setenv("GGKM_ROBUSTNESS_REPLICATIONS", "1")
    with pytest.raises(ValueError, match="at least two"):
        study.build_design("final")


@pytest.mark.parametrize("distribution", ["poisson", "negative_binomial", "binomial"])
def test_known_latency_uses_shared_em_without_changing_generating_parameters(distribution):
    case = {**study.build_experiments()[0], "n": 24}
    data, _ = study.generate_replication(case, 0, study.build_design(replications=2), 3.5)
    X = StandardScaler().fit_transform(data["X"])
    model = runner.KnownLatencyGGKM(
        latent_distribution=distribution, K_bin=2, max_em_iter=2,
        **study.GG_PARAMETERS,
    )
    with pytest.warns(Warning):
        model.fit(X, Surv.from_arrays(data["event"].astype(bool), data["time"]))
    assert (model.a_, model.d_, model.p_) == (1.6, 1.3, 1.1)
    assert not any(row["stage"] == "latency" for row in model.optimizer_results_)
    assert np.isfinite(model.predict_survival(X, [0, 1, 2])).all()


@pytest.fixture
def saved_study(tmp_path, monkeypatch):
    all_cases = study.build_experiments()
    cases = [all_cases[0], all_cases[4], all_cases[40]]  # Poisson n=300/600, mixture n=300.
    cases = [{**case, "task_id": index} for index, case in enumerate(cases)]
    monkeypatch.setattr(study, "build_experiments", lambda: cases)
    monkeypatch.setattr(study, "CALIBRATION_SIZE", 128)
    design = study.build_design(replications=4)
    directory = tmp_path / "results"
    seen = []

    def fake_fit(model, mode, case, design, context, base):
        seen.append((base["task_id"], base["replication"], model, mode, id(context), base["data_hash"]))
        assert context["X_train"].shape[1] == 2  # Never expose latent N to fitting.
        delta = [0.01, 0.03, 0.02, 0.04][base["replication"]] if model == "bn" else (0.05 if model == "binomial" else 0.0)
        if mode == "fixed":
            delta *= 2
        row = {**base, "model_name": model, "latency_mode": mode, "status": "ok", "converged": True,
               "runtime_seconds": 1.0, "integrated_brier": 0.2 + delta, "cure_rmse": 0.1 + delta,
               "survival_rmse": 0.05 + delta, "uno_c_index": 0.8 - delta}
        if model == "bn" and base["replication"] == 1:
            row.update(status="failed", converged=False)
            for metric in study.METRICS:
                row[metric] = 999.0  # Failed finite sentinels must never enter summaries.
        return row

    monkeypatch.setattr(runner, "_fit_one", fake_fit)
    for case in cases:
        runner.run_case(case, design, directory)
    return directory, design, cases, seen


def test_each_replication_reuses_one_observed_dataset_for_every_fit(saved_study):
    _, _, cases, seen = saved_study
    assert len(seen) == 4 * (3 + 6 + 3)
    for case in cases:
        for replication in range(4):
            group = [row for row in seen if row[:2] == (case["task_id"], replication)]
            assert len({row[-1] for row in group}) == len({row[-2] for row in group}) == 1


def test_completed_replications_resume_and_tampered_samples_are_rejected(saved_study, monkeypatch):
    directory, design, cases, _ = saved_study
    monkeypatch.setattr(runner, "_fit_one", lambda *args: pytest.fail("Unexpected refit"))
    runner.run_case(cases[0], design, directory)
    _, archive = runner.replication_paths(directory, 0, 0)
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="modified replication sample"):
        runner.run_case(cases[0], design, directory)


def test_different_study_settings_cannot_reuse_output_directory(saved_study):
    directory, design, _, _ = saved_study
    changed = {**design, "master_seed": design["master_seed"] + 1}
    with pytest.raises(ValueError, match="Different study design"):
        study.write_design(directory, changed)


def test_paired_reports_use_correct_references_finite_pairs_and_metric_direction(saved_study):
    directory, _, _, _ = saved_study
    _, frame, missing = reporting.read_results(directory)
    assert missing == []
    pairs = reporting.paired_replications(frame)
    assert "poisson_mixture" not in set(pairs.truth)
    contrasts = reporting.paired_summary(frame, pairs)
    selected = contrasts.loc[contrasts.task_id.eq(0) & contrasts.model_name.eq("bn")]
    assert set(selected.valid) == {3}
    np.testing.assert_allclose(selected["mean"], np.mean([0.01, 0.02, 0.04]))
    np.testing.assert_allclose(selected.mcse, np.std([0.01, 0.02, 0.04], ddof=1) / np.sqrt(3))
    assert set(selected.specification) == {"poisson_limit"}
    assert set(selected.assessment) == {"not_assessed"}
    assert contrasts.within_margin.isna().all()
    with_margins = reporting.paired_summary(frame, pairs, margins={"integrated_brier": 0.1})
    assert with_margins.loc[with_margins.metric.eq("integrated_brier"), "within_margin"].all()
    compensated = reporting.latency_compensation(pairs)
    nb = compensated.loc[compensated.model_name.eq("bn")]
    np.testing.assert_allclose(nb["mean"], -np.mean([0.01, 0.02, 0.04]))


@pytest.mark.parametrize("damage", ["checksum", "unpaired", "duplicate", "seed", "missing"])
def test_reports_reject_damaged_or_unpaired_case_outputs(saved_study, damage):
    directory, design, _, _ = saved_study
    path = directory / "replications_case_000.csv"
    if damage == "checksum":
        path.write_text(path.read_text() + "\n")
    elif damage == "missing":
        path.unlink()
    else:
        frame = pd.read_csv(path)
        if damage == "unpaired":
            frame.loc[0, "train_index_hash"] = "wrong"
        elif damage == "duplicate":
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        else:
            frame.loc[0, "inner_seed"] += 1
        runner.atomic_write_csv(path, frame)
        metadata_path = path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["csv_sha256"] = study.checksum(path)
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        reporting.read_results(directory)


def test_pilot_allocation_uses_paired_sd_and_observed_finite_pair_fraction(saved_study):
    directory, design, _, _ = saved_study
    _, frame, _ = reporting.read_results(directory)
    contrasts = reporting.paired_summary(frame, reporting.paired_replications(frame))
    projections, report = reporting.pilot_precision(design, contrasts, [], ibs_target=0.001, cure_rmse_target=0.001)
    nb = projections.loc[projections.task_id.eq(0) & projections.model_name.eq("bn")].iloc[0]
    expected = np.std([0.01, 0.02, 0.04], ddof=1)
    assert nb.required_replications == max(2, np.ceil((expected / 0.001) ** 2 / 0.75))
    assert nb.projected_mcse == pytest.approx(expected / np.sqrt(250 * 0.75))
    assert report["decision"] == "insufficient_precision"
    _, missing = reporting.pilot_precision(design, contrasts, [0])
    assert missing["decision"] == "incomplete"


def test_tables_and_heatmaps_include_stress_without_claiming_a_matched_reference(saved_study, tmp_path):
    directory, _, _, _ = saved_study
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    performance, contrasts = reporting.generate_report(results_dir=directory, output_dir=tables)
    assert set(performance.study_part) == {"primary", "stress"}
    assert set(contrasts.study_part) == {"primary"}
    text = (tables / "latent-robustness-performance-table.tex").read_text()
    assert "Poisson mixture" in text
    assert "Monte Carlo standard errors" in text
    assert not re.search(r"\b(?:nan|inf)\b", text, re.IGNORECASE)
    assert (tables / "paired_replications.csv").is_file()
    report = json.loads((tables / "report.json").read_text())
    assert report["practical_margins"] == {}
    paths = reporting.generate_report(results_dir=directory, output_dir=figures, kind="figures")
    assert len(paths) == 36
    assert {path.suffix for path in paths} == {".pdf", ".png", ".svg"}
    assert all(path.stat().st_size > 100 for path in paths)


def test_robustness_pbs_jobs_have_their_own_dependency_chain():
    jobs = select_jobs(["latent-robustness"])
    assert [job.stage for job in jobs] == ["pilot", "pilot-summary", "experiments", "figures", "tables"]
    assert [job.prerequisite for job in jobs] == [None, jobs[0].key, jobs[1].key, jobs[2].key, jobs[2].key]
    for job in jobs:
        source = (study.PROJECT_ROOT / "pbs_jobs" / job.pbs_file).read_text()
        assert f"ggkm_run_job {job.key}" in source
        if job.is_array:
            assert "#PBS -J 0-47" in source
    assert [job.stage for job in select_jobs(["latent-robustness"], "pilot")] == ["pilot", "pilot-summary"]
