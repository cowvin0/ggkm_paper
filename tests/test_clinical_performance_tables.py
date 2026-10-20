"""Exercise clinical table regeneration using synthetic paired holdout CSVs."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.create_clinical_performance_tables import generate_tables, main
from scripts.table_utils.clinical import ClinicalMethod, ClinicalResults, methods, render_performance_table


METHOD = ClinicalMethod("mixture_cure")


def _frame(method=METHOD):
    return pd.DataFrame({
        "split": [2, 0, 1, 3],
        "model_name": [method.model] * 4,
        "kernel": [method.kernel] * 4,
        "status": ["ok", "ok", "failed", "ok"],
        "converged": ["False", "True", "True", "1"],
        "integrated_brier": [0.1, 0.3, 999.0, np.nan],
        "cure_auc": [0.9, 0.7, 999.0, 0.8],
        "mean_predicted_cure": [0.2, 0.4, 999.0, 0.6],
        "runtime_seconds": [1.0, 2.0, 1000.0, 4.0],
        "train_index_hash": ["train2", "train0", "train1", "train3"],
        "test_index_hash": ["test2", "test0", "test1", "test3"],
        "inner_seed": [1002, 1000, 1001, 1003],
        "training_size": [143] * 4,
        "test_size": [62] * 4,
    })


def _write(directory, *, dataset="melanoma", method=METHOD, frame=None, task=0):
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"_kernel_{method.kernel}" if method.kernel != "parametric" else ""
    path = directory / f"splits_{dataset.replace('-', '_')}_task{task}_{method.model}{suffix}.csv"
    (frame if frame is not None else _frame(method)).to_csv(path, index=False)
    return path


def _complete(directory, *, dataset="melanoma", include_sigmoid=False):
    for task, method in enumerate(methods(include_sigmoid=include_sigmoid)):
        _write(directory, dataset=dataset, method=method, task=task)
    return ClinicalResults(directory, dataset, include_sigmoid=include_sigmoid)


def test_summary_uses_successful_finite_values_and_sample_sd(tmp_path):
    _write(tmp_path)
    results = ClinicalResults(tmp_path, "melanoma")

    mean, sd = results.summary(METHOD, "integrated_brier")
    assert mean == pytest.approx(0.2)
    assert sd == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert results.summary(METHOD, "mean_predicted_cure") == pytest.approx((0.4, 0.2))
    assert results.convergence(METHOD) == (2, 3)
    assert results.runtime(METHOD) == pytest.approx((7 / 3, 4))
    assert results.frame(METHOD)["split"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("values, expected", [
    ([np.inf, np.nan, 999, -np.inf], None), ([0.2, np.nan, 999, np.inf], 0.2),
])
def test_nonfinite_and_singleton_metrics_have_undefined_sd(tmp_path, values, expected):
    frame = _frame()
    frame["integrated_brier"] = values
    _write(tmp_path, frame=frame)
    mean, sd = ClinicalResults(tmp_path, "melanoma").summary(METHOD, "integrated_brier")

    assert np.isnan(sd)
    assert np.isnan(mean) if expected is None else mean == expected


@pytest.mark.parametrize("column", [
    "split", "status", "integrated_brier", "cure_auc", "mean_predicted_cure", "converged",
    "runtime_seconds", "train_index_hash", "test_index_hash",
])
def test_missing_essential_columns_fail_clearly(tmp_path, column):
    _write(tmp_path, frame=_frame().drop(columns=column))
    with pytest.raises(ValueError, match=f"missing required columns: {column}"):
        ClinicalResults(tmp_path, "melanoma").frame(METHOD)


@pytest.mark.parametrize("values", [[2, 0, 1, 2], [2, np.nan, 1, 3], [2, -1, 1, 3], [2, 0.5, 1, 3]])
def test_invalid_or_duplicate_split_ids_fail(tmp_path, values):
    frame = _frame()
    frame["split"] = values
    _write(tmp_path, frame=frame)
    with pytest.raises(ValueError, match="split IDs"):
        ClinicalResults(tmp_path, "melanoma").frame(METHOD)


@pytest.mark.parametrize("column, value, message", [
    ("status", "success", "status must be explicitly"),
    ("converged", "unknown", "invalid convergence value"),
    ("converged", np.nan, "missing convergence status"),
    ("runtime_seconds", -1, "must be nonnegative"),
    ("cure_auc", "bad", "contains nonnumeric"),
    ("train_index_hash", "", "missing train_index_hash"),
    ("test_index_hash", np.nan, "missing test_index_hash"),
    ("model_name", "poisson", "metadata does not match the filename"),
    ("kernel", "rbf", "metadata does not match the filename"),
])
def test_bad_values_fail_clearly(tmp_path, column, value, message):
    frame = _frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    _write(tmp_path, frame=frame)
    with pytest.raises(ValueError, match=message):
        ClinicalResults(tmp_path, "melanoma").frame(METHOD)


@pytest.mark.parametrize("values", [
    ["False", "True", "False", "True"], ["0", "1", "0", "1"],
    ["0.0", "1.0", "0.0", "1.0"], [False, True, False, True],
])
def test_convergence_parses_false_without_string_truthiness(tmp_path, values):
    frame = _frame()
    frame["converged"] = values
    _write(tmp_path, frame=frame)
    assert ClinicalResults(tmp_path, "melanoma").convergence(METHOD) == (2, 3)


def test_duplicate_configurations_are_rejected_and_unselected_files_ignored(tmp_path):
    _write(tmp_path)
    (tmp_path / "splits_melanoma_task1_bernoulli_kernel_rbf.csv").write_text("unused malformed input\n")
    (tmp_path / "summary_melanoma_task0_mixture_cure.csv").write_text("unused summary\n")
    assert len(ClinicalResults(tmp_path, "melanoma").frame(METHOD)) == 4
    _write(tmp_path, task=99)
    with pytest.raises(ValueError, match="Duplicate melanoma split CSVs"):
        ClinicalResults(tmp_path, "melanoma")


def test_duplicate_headers_and_empty_csvs_are_rejected(tmp_path):
    path = _write(tmp_path)
    path.write_text(path.read_text().replace("split,model_name", "split,split", 1))
    with pytest.raises(ValueError, match="duplicate CSV column names"):
        ClinicalResults(tmp_path, "melanoma").frame(METHOD)
    _write(tmp_path, frame=_frame().iloc[:0])
    with pytest.raises(ValueError, match="split CSV has no rows"):
        ClinicalResults(tmp_path, "melanoma").frame(METHOD)


def test_loading_discards_unneeded_large_payloads(tmp_path):
    frame = _frame()
    frame["best_params"] = "large JSON string"
    frame["plot_data_file"] = "unneeded prediction file"
    _write(tmp_path, frame=frame)
    loaded = ClinicalResults(tmp_path, "melanoma").frame(METHOD)
    assert "best_params" not in loaded
    assert "plot_data_file" not in loaded


def test_complete_validation_infers_planned_splits_and_keeps_failed_fits(tmp_path):
    results = _complete(tmp_path)
    assert results.validate() == 4
    assert results.validate(4) == 4
    assert len(results.methods) == 18
    with pytest.raises(ValueError, match="expected 50 planned splits"):
        results.validate(50)
    with pytest.raises(ValueError, match="positive integer"):
        results.validate(0)


def test_incomplete_configuration_missing_method_and_shifted_ids_fail(tmp_path):
    results = _complete(tmp_path)
    path = results.path(METHOD)
    _write(tmp_path, frame=_frame().iloc[:3])
    with pytest.raises(ValueError, match="inconsistent numbers of attempted holdouts"):
        ClinicalResults(tmp_path, "melanoma").validate()
    frame = _frame()
    frame["split"] += 1
    _write(tmp_path, frame=frame)
    with pytest.raises(ValueError, match="expected split IDs 0 through 3"):
        ClinicalResults(tmp_path, "melanoma").validate()
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing melanoma split CSV for mixture_cure/parametric"):
        ClinicalResults(tmp_path, "melanoma").validate()


@pytest.mark.parametrize("column", ["train_index_hash", "test_index_hash", "inner_seed", "training_size", "test_size"])
def test_same_split_must_be_paired_across_every_method(tmp_path, column):
    results = _complete(tmp_path)
    method = ClinicalMethod("bn", "wavelet")
    path = results.path(method)
    frame = pd.read_csv(path)
    frame.loc[0, column] = "different" if column.endswith("hash") else 999
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=f"{column} differs between methods"):
        ClinicalResults(tmp_path, "melanoma").validate()


def test_render_preserves_labels_columns_order_precision_and_sd(tmp_path):
    results = _complete(tmp_path)
    text = render_performance_table(results, planned_splits=results.validate())

    assert r"\label{tab:melanoma-performance}" in text
    assert r"\begin{tabular}{llccccc}" in text
    assert "4 paired repeated holdouts" in text
    assert "sample split standard deviations" in text
    assert "Runtime is the mean across successful fits" in text
    assert text.count("& 2/3\n& 0.2000 (0.1414)\n& 0.8000 (0.1000)\n& 0.400 (0.200)\n& 2.33 (4.00)") == 18
    assert text.index("Mixture cure\n& Mixture cure") < text.index("Promotion time\n& Poisson")
    assert text.index("Spline\n& Negative binomial") < text.index("GG-KM (linear)")
    assert text.index("GG-KM (Cauchy)") < text.index("GG-KM (polynomial)") < text.index("GG-KM (wavelet)")
    assert "sigmoid" not in text
    assert text.count(r"\hline") == 5


def test_undefined_summaries_render_as_dashes(tmp_path):
    results = _complete(tmp_path)
    frame = _frame()
    frame["status"] = "failed"
    _write(tmp_path, frame=frame)
    text = render_performance_table(ClinicalResults(tmp_path, "melanoma"), planned_splits=4)
    assert "& 0/0\n& --\n& --\n& --\n& -- " in text
    assert "nan" not in text
    frame.loc[0, "status"] = "ok"
    _write(tmp_path, frame=frame)
    text = render_performance_table(ClinicalResults(tmp_path, "melanoma"), planned_splits=4)
    assert "& 0.1000 (--)" in text


def test_sigmoid_is_optional_only_for_breast_cancer(tmp_path):
    _complete(tmp_path, dataset="breast-cancer", include_sigmoid=True)
    standard = ClinicalResults(tmp_path, "breast-cancer")
    extended = ClinicalResults(tmp_path, "breast-cancer", include_sigmoid=True)
    assert standard.validate() == extended.validate() == 4
    assert len(standard.methods) == 18
    assert len(extended.methods) == 20
    text = render_performance_table(extended, planned_splits=4)
    assert text.count("GG-KM (sigmoid)") == 2
    assert text.index("GG-KM (Cauchy)") < text.index("GG-KM (sigmoid)") < text.index("GG-KM (polynomial)")
    assert r"\label{tab:breast-cancer-performance}" in text
    assert "Paraíba State, Brazil" in text
    with pytest.raises(ValueError, match="only to breast cancer"):
        ClinicalResults(tmp_path, "melanoma", include_sigmoid=True)


def test_generator_writes_both_tables_and_replaces_old_values(tmp_path):
    melanoma = tmp_path / "melanoma"
    breast = tmp_path / "breast"
    output = tmp_path / "tables"
    _complete(melanoma)
    _complete(breast, dataset="breast-cancer")
    output.mkdir()
    (output / "melanoma-performance-table.tex").write_text("old data")

    paths = generate_tables(
        melanoma_results_dir=melanoma, breast_cancer_results_dir=breast,
        output_dir=output, planned_splits=4,
    )

    assert {path.name for path in paths} == {"melanoma-performance-table.tex", "breast-cancer-performance-table.tex"}
    for path in paths:
        assert "old data" not in path.read_text()
        assert "0.2000 (0.1414)" in path.read_text()
    assert not list(output.glob("*.tmp"))


def test_error_in_second_cohort_preserves_every_existing_table(tmp_path):
    melanoma = tmp_path / "melanoma"
    breast = tmp_path / "breast"
    output = tmp_path / "tables"
    _complete(melanoma)
    results = _complete(breast, dataset="breast-cancer")
    results.path(ClinicalMethod("bn", "wavelet")).unlink()
    output.mkdir()
    for name in ("melanoma-performance-table.tex", "breast-cancer-performance-table.tex"):
        (output / name).write_text("original table")

    with pytest.raises(FileNotFoundError, match="Missing breast-cancer split CSV"):
        generate_tables(
            melanoma_results_dir=melanoma, breast_cancer_results_dir=breast, output_dir=output,
        )
    assert all(path.read_text() == "original table" for path in output.iterdir())


def test_single_dataset_cli_needs_only_its_own_results(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "tables"
    _complete(results, dataset="breast-cancer", include_sigmoid=True)

    main([
        "--dataset", "breast-cancer", "--breast-cancer-results-dir", str(results),
        "--output-dir", str(output), "--planned-splits", "4", "--include-sigmoid",
    ])

    assert [path.name for path in output.iterdir()] == ["breast-cancer-performance-table.tex"]
    assert (output / "breast-cancer-performance-table.tex").read_text().count("GG-KM (sigmoid)") == 2


def test_cli_missing_directory_has_actionable_error_and_no_outputs(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([
            "--dataset", "melanoma", "--melanoma-results-dir", str(tmp_path / "missing"),
            "--output-dir", str(tmp_path / "tables"),
        ])
    assert error.value.code == 2
    assert "Run on the cluster" in capsys.readouterr().err
    assert not (tmp_path / "tables").exists()


def test_direct_cli_invocation_works_outside_project(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_clinical_performance_tables.py"
    process = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    assert "--breast-cancer-results-dir" in process.stdout
    assert "--planned-splits" in process.stdout
