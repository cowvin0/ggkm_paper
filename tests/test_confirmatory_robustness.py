"""Numerical controls, prospective gates, paired inference, and resumability."""

import copy
import json
import os
import shlex
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

from skcure._latent import _NegativeBinomialLatentModel, _PoissonLatentModel
from skcure.distributions import GeneralizedGamma
from scripts import confirmatory_robustness as runner
from scripts import confirmatory_robustness_fitting as fitting
from scripts import confirmatory_robustness_utils as confirm
from scripts import confirmatory_robustness_storage as storage
from scripts import latent_robustness_utils as original
from scripts.experiment_utils.artifacts import atomic_write_json
from scripts.run_confirmatory_workflow import jobs_for


@pytest.fixture
def config():
    config = copy.deepcopy(confirm.load_protocol())
    config["storage"] = {"format": "full"}  # Retain coverage of legacy outputs.
    config.update(margins={"integrated_brier": 0.02, "cure_rmse": 0.03},
                  margin_justification="Synthetic test tolerances, not scientific defaults",
                  audit_replications=2, pilot_replications=4, minimum_final_replications=4)
    return config


def test_poisson_limit_holds_for_survival_cure_and_likelihood():
    latency = GeneralizedGamma(a=1.6, d=1.3, p=1.1)
    poisson, nb = _PoissonLatentModel(), _NegativeBinomialLatentModel(phi=1e-10)
    eta = np.array([-1., 0., 1.])
    times = np.array([0., 0.5, 2., 10.])
    np.testing.assert_allclose(nb.predict_survival_from_eta(latency, eta, times), poisson.predict_survival_from_eta(latency, eta, times), atol=1e-9)
    np.testing.assert_allclose(nb.predict_cure_probability_from_eta(latency, eta), poisson.predict_cure_probability_from_eta(latency, eta), atol=1e-9)
    observed, event = np.array([.5, 1., 2.]), np.array([1, 0, 1])
    assert nb.observed_loglikelihood_from_eta(latency, eta, observed, event) == pytest.approx(poisson.observed_loglikelihood_from_eta(latency, eta, observed, event), abs=1e-8)


def test_estimated_phi_initialization_does_not_fix_phi_and_fixed_latency_stays_fixed():
    case = {**original.build_experiments()[0], "n": 24}
    data, _ = original.generate_replication(case, 0, original.build_design(replications=2), 3.5)
    X = StandardScaler().fit_transform(data["X"])
    model = fitting.InitializedGGKM(latent_distribution="negative_binomial", a=1.6, d=1.3, p=1.1, max_em_iter=2)
    model.dispersion_start, model.freeze_latency = 2., True
    model._latent_model = model._make_latent_model(model._NB, None, 1)
    model._reset_parameters_for_fit()
    assert model._latent_model.phi == 2.
    assert model.estimate_phi is True
    with pytest.warns(Warning):
        model.fit(X, Surv.from_arrays(data["event"].astype(bool), data["time"]))
    assert model.estimate_phi is True
    assert (model.a_, model.d_, model.p_) == (1.6, 1.3, 1.1)
    assert not any(row["stage"] == "latency" for row in model.optimizer_results_)
    assert np.isfinite(model.final_loglik_)


def test_multistart_retries_and_selects_likelihood_not_test_metrics(monkeypatch, config):
    calls = []
    config["fitting"]["profile_restart"] = False  # Isolate the prescribed cold starts.

    class FakeModel:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, X, time, event):
            calls.append((self.dispersion_start, self.parameters["max_em_iter"]))
            self.converged_ = self.parameters["max_em_iter"] >= 20
            self.a_, self.d_, self.p_ = (self.parameters[key] for key in ("a", "d", "p"))
            self.phi_ = self.dispersion_start
            self.final_loglik_ = self.dispersion_start + self.a_
            self.loglik_history_ = [-10., self.final_loglik_]
            self.optimizer_results_ = [{"objective": np.inf, "status": np.int64(0)}]
            self.termination_reason_ = "converged" if self.converged_ else "max_em_iter"
            self.n_iter_ = self.parameters["max_em_iter"]
            return self

    monkeypatch.setattr(fitting, "InitializedGGKM", FakeModel)
    fit = fitting.MultiStartGGKM(fitting=config["fitting"], latent_distribution="negative_binomial", phi=None, max_em_iter=10)
    fit.fit(np.zeros((4, 2)), np.ones(4), np.ones(4))
    assert len(calls) == 16  # 2 latency starts * 4 dispersion starts * 2 attempts.
    assert fit.model_.phi_ == 2.
    assert fit.model_.a_ == 1.6
    assert fit.model_.converged_
    assert len(fit.attempts_) == 16
    json.dumps(fit.attempts_, allow_nan=False)


def test_profile_reoptimizes_at_fixed_phi_and_includes_exact_poisson(monkeypatch, config):
    received = []

    class FakeFit:
        def __init__(self, **parameters):
            received.append(parameters)
            self.attempts_ = []

        def fit(self, X, time, event):
            self.model_ = type("Model", (), dict(final_loglik_=-10., converged_=True, a_=1., d_=1., p_=1.))()

    monkeypatch.setattr(fitting, "MultiStartGGKM", FakeFit)
    rows = fitting.profile_dispersion({"gamma": .25, "lambda_reg": .3}, config["fitting"], None, None, None)
    assert len(rows) == 7
    assert received[0]["latent_distribution"] == "poisson"
    assert received[0]["phi"] is None
    assert all(row["phi"] > 0 for row in received[1:])
    assert all(row["gamma"] == .25 and row["lambda_reg"] == .3 and row["fixed_latency"] is None for row in received)


@pytest.mark.parametrize("phi", [0., .5])
@pytest.mark.parametrize("freeze_latency", [False, True])
def test_profile_continuation_preserves_full_state_and_frees_phi(monkeypatch, phi, freeze_latency):
    case = {**original.build_experiments()[28], "n": 32}
    data, _ = original.generate_replication(case, 0, original.build_design(replications=2), 3.5)
    X = StandardScaler().fit_transform(data["X"])
    seed = fitting.InitializedGGKM(latent_distribution="poisson" if phi == 0 else "negative_binomial",
                                   phi=None if phi == 0 else phi, a=1.6, d=1.3, p=1.1, max_em_iter=3)
    seed.freeze_latency, seed.dispersion_start = freeze_latency, None
    with warnings.catch_warnings(action="ignore"):
        seed.fit(X, Surv.from_arrays(data["event"].astype(bool), data["time"]))
    coefficients, seed_ll = seed.alpha_.copy(), seed.final_loglik_
    assert np.linalg.norm(coefficients) > 0  # A reset to zeros would change this start.
    original_run_em = fitting.run_em

    def inspect_start(parameters, **kwargs):
        assert parameters[0] == seed.beta0_
        np.testing.assert_array_equal(parameters[1], coefficients)
        assert kwargs["state"].model.estimate_phi
        assert not kwargs["state"].model.converged_  # Never inherit fixed-profile convergence.
        assert kwargs["state"].model.phi is None
        np.testing.assert_array_equal(kwargs["state"].model.K_, seed.K_)
        assert [kwargs["state"].model.a_, kwargs["state"].model.d_, kwargs["state"].model.p_] == [seed.a_, seed.d_, seed.p_]
        return original_run_em(parameters, **kwargs)

    monkeypatch.setattr(fitting, "run_em", inspect_start)
    with warnings.catch_warnings(action="ignore"):
        resumed = fitting._resume_free_nb(seed, phi, data["time"], data["event"], 8)
    assert resumed.final_loglik_ >= resumed.continuation_initial_loglik_ - 1e-8
    assert resumed.estimate_phi
    assert resumed.phi_ > 0
    if freeze_latency:
        assert (resumed.a_, resumed.d_, resumed.p_) == (1.6, 1.3, 1.1)
    assert any(row["stage"] == "latency" for row in resumed.optimizer_results_) != freeze_latency
    np.testing.assert_array_equal(seed.alpha_, coefficients)
    assert seed.final_loglik_ == seed_ll
    assert seed.latent_distribution == ("poisson" if phi == 0 else "negative_binomial")

    def worsened_dispersion(steps, parameters, expected):
        steps.model._latent_model.phi = 1e12
        return parameters

    monkeypatch.setattr(fitting._GGKMSteps, "maximize_dispersion", worsened_dispersion)
    with pytest.raises(RuntimeError, match="decreased the penalized objective"):
        fitting._resume_free_nb(seed, phi, data["time"], data["event"], 8)
    np.testing.assert_array_equal(seed.alpha_, coefficients)
    assert seed.final_loglik_ == seed_ll


def test_profile_restart_is_selected_only_after_a_real_free_fit(monkeypatch, config):
    config["fitting"].update(latency_starts=[[1., 1., 1.]], dispersion_starts=[1.], retries=0)

    class FakeModel:
        def __init__(self, **parameters):
            self.a_ = self.d_ = self.p_ = 1.
            self.phi_, self.final_loglik_ = 1., -12.
            self.converged_, self.n_iter_ = True, 1
            self.termination_reason_, self.loglik_history_, self.optimizer_results_ = "converged", [-12.], []

        def fit(self, X, time, event):
            return self

    seed = FakeModel()
    seed.beta0_, seed.alpha_, seed.final_loglik_ = 2., np.array([.1, .2]), -10.
    calls = []

    def resume(model, phi, time, event, budget):
        calls.append((model, phi))
        result = FakeModel()
        result.final_loglik_, result.continuation_initial_loglik_ = -9., -10.
        result.loglik_history_ = [-9.]
        return result

    monkeypatch.setattr(fitting, "InitializedGGKM", FakeModel)
    monkeypatch.setattr(fitting, "profile_dispersion", lambda *args, **kwargs: ([{"phi": .5}], [(.5, seed)]))
    monkeypatch.setattr(fitting, "_resume_free_nb", resume)
    fit = fitting.MultiStartGGKM(fitting=config["fitting"], latent_distribution="negative_binomial", max_em_iter=10)
    fit.fit(None, None, None)
    assert calls == [(seed, .5)]
    assert fit.model_.final_loglik_ == -9.
    assert fit.attempts_[fit.selected_attempt_]["source"] == "profile_restart"
    assert fit.attempts_[fit.selected_attempt_]["initial_loglik"] == -10.
    json.dumps(fit.attempts_, allow_nan=False)

    def failed_resume(*args):
        raise RuntimeError("Continuation did not succeed")

    monkeypatch.setattr(fitting, "_resume_free_nb", failed_resume)
    fit.fit(None, None, None)
    assert fit.model_.final_loglik_ == -12.  # The better fixed profile is never a free-fit fallback.
    assert fit.attempts_[-1]["status"] == "failed"
    assert not fit.attempts_[-1]["converged"]


@pytest.mark.parametrize("model,mode", [("poisson", "estimated"), ("bn", "estimated"), ("bn", "fixed"), ("bn", "phi_fixed"), ("binomial", "estimated")])
def test_real_inner_tuning_refit_scoring_and_diagnostic_serialization(tmp_path, config, model, mode):
    config["fitting"].update(latency_starts=[[1.6, 1.3, 1.1]], dispersion_starts=[1.],
                             profile_dispersion=[0., .5], retries=0)
    design = runner.build_design("audit", tmp_path, config)
    design["stage"] = "pilot"  # The numerical smoke test omits the expensive audit profile.
    design["truth_grid_points"] = 3
    design["inner_grid_points"] = design["outer_grid_points"] = 4
    for name in original.FIT_MODELS:
        design["estimator_parameters"][name]["max_em_iter"] = 3
        design["parameter_grids"][name] = design["parameter_grids"][name][:1]
    case = {**original.build_experiments()[28], "n": 48}
    context, base, _ = runner.prepare_replication(case, 0, design, {"censor_mean": 3.5})
    row = runner.fit_one(model, mode, case, design, context, base, tmp_path)
    assert row["status"] == "ok", row["failure_reason"]
    assert np.isfinite(row["integrated_brier"])
    assert np.isfinite(row["cure_rmse"])
    diagnostic = json.loads((tmp_path / row["diagnostic_path"]).read_text())
    assert len(diagnostic["fits"]) == 3  # Two inner folds and the held-out refit.
    assert all(fit["attempts"] for fit in diagnostic["fits"])
    if model == "bn" and mode != "phi_fixed":
        assert all(fit["profile"] for fit in diagnostic["fits"])
        assert all(any(attempt.get("source") == "profile_restart" for attempt in fit["attempts"])
                   for fit in diagnostic["fits"])
    if mode == "fixed":
        assert [row[key] for key in ("fitted_a", "fitted_d", "fitted_p")] == [1.6, 1.3, 1.1]
    if mode == "phi_fixed":
        assert row["fitted_phi"] == 2.


def synthetic_frame():
    rows = []
    for case in original.build_experiments():
        if case["study_part"] != "primary":
            continue
        for replication in range(4):
            for model in original.FIT_MODELS:
                delta = 0. if model == case["reference_model"] else .001 * (replication + 1)
                rows.append({**case, "replication": replication, "model_name": model, "latency_mode": "estimated",
                             "status": "ok", "converged": True, "integrated_brier": .2 + delta,
                             "cure_rmse": .1 + delta, "survival_rmse": .1 + delta, "uno_c_index": .8 - delta})
    return pd.DataFrame(rows)


def test_simultaneous_bounds_cover_all_primary_comparisons_and_correct_direction(config):
    frame = synthetic_frame()
    pairs = confirm.paired_values(frame)
    summary = confirm.contrast_summary(frame, pairs, config)
    assert summary.primary.sum() == confirm.FAMILY_SIZE == 320
    row = summary.loc[summary.primary].iloc[0]
    expected = np.std([.001, .002, .003, .004], ddof=1) / 2
    assert row.mcse == pytest.approx(expected)
    assert row.simultaneous_upper == pytest.approx(.0025 + student_t.ppf(1 - .05 / 320, 3) * expected)
    np.testing.assert_allclose(summary["mean"], .0025)
    assert not summary.loc[summary.metric.eq("uno_c_index"), "primary"].any()
    assert "poisson_limit" in set(summary.specification)
    assert "poisson_mixture" not in set(summary.truth)


def test_failures_and_nonconvergence_cannot_disappear_from_decision(config):
    frame = synthetic_frame()
    mask = frame.task_id.eq(0) & frame.model_name.eq("bn") & frame.replication.eq(0)
    frame.loc[mask, "converged"] = False
    summary = confirm.contrast_summary(frame, confirm.paired_values(frame), config)
    selected = summary.loc[summary.task_id.eq(0) & summary.model_name.eq("bn") & summary.metric.eq("integrated_brier")]
    assert list(selected.valid) == [4, 3]
    frame.loc[mask, "status"] = "failed"
    summary = confirm.contrast_summary(frame, confirm.paired_values(frame), config)
    assert set(summary.loc[summary.task_id.eq(0) & summary.model_name.eq("bn") & summary.primary & summary.analysis.eq("all_finite"), "assessment"]) == {"incomplete_pairs"}


def test_no_finite_pairs_produces_incomplete_assessments_and_reviewable_artifacts(tmp_path, config):
    frame = synthetic_frame()
    frame["status"] = "failed"
    summary = confirm.contrast_summary(frame, confirm.paired_values(frame), config)
    assert set(summary.loc[summary.primary, "assessment"]) == {"incomplete_pairs"}
    sensitivity = confirm.margin_sensitivity(summary)
    assert not sensitivity.all_within.any()
    assert sensitivity.finite_bounds.sum() == 0
    report = {"decision": "incomplete", "margins": config["margins"], "margin_justification": config["margin_justification"]}
    runner.write_summary_artifacts(tmp_path, summary, report)
    assert (tmp_path / "conclusion.md").is_file()
    assert (tmp_path / "confirmatory_cure_rmse.pdf").is_file()


def test_margin_sensitivity_is_descriptive_and_does_not_change_protocol(config):
    before = copy.deepcopy(config)
    frame = synthetic_frame()
    summary = confirm.contrast_summary(frame, confirm.paired_values(frame), config)
    review = confirm.margin_sensitivity(summary)
    assert len(review) == 18  # Nine illustrative margins, each in both analyses.
    assert (review.within_margin <= review.comparisons).all()
    assert config == before
    historical = summary.loc[summary.analysis.eq("all_finite") & summary.primary].copy()
    historical["analysis"] = "historical_all_finite"
    assert len(confirm.margin_sensitivity(historical)) == 9


def test_precision_allocation_uses_paired_sd_and_simultaneous_margin_scaled_halfwidth(config):
    frame = synthetic_frame()
    summary = confirm.contrast_summary(frame, confirm.paired_values(frame), config)
    projections, report = confirm.allocation(summary, config)
    assert len(projections) == 320
    assert report["ready"]
    for row in projections.itertuples():
        count = int(np.floor(report["final_replications"] * row.finite_pair_fraction))
        achieved = student_t.ppf(1 - .05 / 320, count - 1) * row.sd / np.sqrt(count)
        assert achieved <= row.target_halfwidth
    config["final_replications"] = 2
    _, report = confirm.allocation(summary, config)
    assert not report["ready"]


def test_unset_margins_do_not_prevent_audit_but_block_confirmation(tmp_path):
    config = confirm.load_protocol()
    config["margins"] = dict.fromkeys(confirm.PRIMARY_METRICS)
    config["margin_justification"] = None
    assert all(value is None for value in config["margins"].values())
    runner.build_design("audit", tmp_path, config)
    with pytest.raises(ValueError, match="scientific"):
        runner.build_design("pilot", tmp_path, config)


def test_margins_can_be_chosen_after_audit_but_not_after_pilot_freeze(tmp_path, config):
    unset = copy.deepcopy(config)
    unset["margins"] = dict.fromkeys(confirm.PRIMARY_METRICS)
    first = runner.build_design("audit", tmp_path, unset)
    second = runner.build_design("audit", tmp_path, config)
    assert first == second
    confirm.freeze_protocol(tmp_path, config)
    config["margins"]["integrated_brier"] *= 2
    with pytest.raises(ValueError, match="immutable"):
        confirm.freeze_protocol(tmp_path, config)


@pytest.fixture
def completed_study(tmp_path, monkeypatch, config, request):
    if getattr(request, "param", "full") == "compact-v1":
        config["storage"] = dict(format="compact-v1", budget_gib=80, projection_factor=2.)
    cases = [original.build_experiments()[index] for index in (4, 28, 40)]
    monkeypatch.setattr(original, "build_experiments", lambda: cases)
    monkeypatch.setattr(original, "CALIBRATION_SIZE", 128)
    monkeypatch.setattr(confirm, "AUDIT_CASE_IDS", (4, 28))
    monkeypatch.setattr(confirm, "FAMILY_SIZE", 16)

    def fake_fit(model, mode, case, design, context, base, directory):
        assert context["X_train"].shape[1] == 2
        delta = 0 if model == case["reference_model"] else .0001 * (base["replication"] % 4 + 1)
        row = {**base, "latency_mode": mode, "model_name": model, "status": "ok", "converged": True,
               "runtime_seconds": .1, "integrated_brier": .2 + delta, "cure_rmse": .1 + delta,
               "survival_rmse": .1 + delta, "uno_c_index": .8 - delta, "final_loglik": -10.,
               "fitted_phi": .001 if model == "bn" else None}
        diagnostic = {"study_id": design["study_id"], "profile": [], "fits": []}
        if design["stage"] == "audit" and model == "bn" and mode != "phi_fixed":
            diagnostic["profile"] = [{"phi": phi, "status": "ok", "converged": True, "final_loglik": -10.1}
                                     for phi in config["fitting"]["profile_dispersion"]]
        path = directory / "diagnostics" / f"{case['task_id']}_{base['replication']}_{mode}_{model}.json"
        if storage.compact(design):
            path = path.with_suffix(".json.gz")
            storage.write_json(path, storage.diagnostic_payload(diagnostic, design))
        else:
            atomic_write_json(path, diagnostic)
        row.update(diagnostic_path=str(path.relative_to(directory)), diagnostic_sha256=original.checksum(path))
        return row

    monkeypatch.setattr(runner, "fit_one", fake_fit)
    for stage in ("audit", "pilot", "final"):
        design = runner.build_design(stage, tmp_path, config)
        for case in design["cases"]:
            runner.run_case(case, design, tmp_path / stage, workers=1)
        if stage == "audit":
            assert runner.audit_summary(tmp_path, config, figures=False)["ready"]
        elif stage == "pilot":
            assert runner.summarize(stage, tmp_path, config, figures=False)["ready"]
    return tmp_path, config


def test_independent_pipeline_with_diagnostics_tables_and_plots(completed_study):
    directory, config = completed_study
    seeds = [json.loads((directory / stage / "design.json").read_text())["master_seed"] for stage in ("audit", "pilot", "final")]
    assert len(set(seeds)) == 3
    report = runner.summarize("final", directory, config)
    assert report["decision"] == "within_prespecified_margins"
    assert not report["sensitivity_disagreement"]
    assert (directory / "final/confirmatory_integrated_brier.pdf").is_file()
    assert (directory / "final/confirmatory_cure_rmse.pdf").is_file()
    assert (directory / "final/fixed_dispersion_controls.csv").stat().st_size > 10
    assert (directory / "final/conclusion.md").is_file()
    assert (directory / "final/confirmatory-contrasts.tex").is_file()


def test_modified_samples_block_reporting_and_resumption(completed_study):
    directory, config = completed_study
    _, archive = runner.replication_paths(directory / "final", 4, 0)
    archive.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Modified sample"):
        runner.read_results("final", directory, config)


def test_profile_improvement_blocks_pilot(completed_study):
    directory, config = completed_study
    report_path = directory / "audit/report.json"
    report = json.loads(report_path.read_text())
    report["ready"] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not ready"):
        runner.build_design("pilot", directory, config)


def test_dependency_chain_and_pbs_resources():
    jobs = jobs_for("all")
    assert [job.stage for job in jobs] == ["audit", "audit-summary", "pilot", "pilot-summary", "final", "report"]
    for previous, current in zip(jobs, jobs[1:]):
        assert current.prerequisite == previous.key
    for job in jobs:
        source = (original.PROJECT_ROOT / "pbs_jobs" / job.pbs_file).read_text()
        assert f"ggkm_run_job {job.key}" in source
        if job.is_array:
            assert f"#PBS -J 0-{job.count-1}" in source


def test_launcher_dry_run_exports_protocol_and_orders_full_pbs_chain(tmp_path, config):
    path = tmp_path / "protocol with spaces.json"
    path.write_text(json.dumps(config))
    target = tmp_path / "fresh results"
    result = subprocess.run([sys.executable, "scripts/run_confirmatory_workflow.py", "--mode", "cluster", "--stage", "all",
                             "--protocol", str(path), "--output-dir", str(target), "--dry-run"],
                            cwd=original.PROJECT_ROOT, capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GGKM_WORKERS": "2"})
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 6
    assert commands[0][commands[0].index("-J") + 1] == "0-5"
    assert commands[2][commands[2].index("-J") + 1] == "0-47"
    assert commands[4][commands[4].index("-J") + 1] == "0-47"
    assert "GGKM_CONFIRMATORY_PROTOCOL" in commands[0][commands[0].index("-v") + 1]
    for previous, command in zip(jobs_for("all"), commands[1:]):
        assert f"depend=afterok:<{previous.key}-job-id>" in command
    assert not target.exists()
