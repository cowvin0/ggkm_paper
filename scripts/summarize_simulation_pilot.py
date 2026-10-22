#!/usr/bin/env python3
"""Calculate precision requirements from the independent primary simulation pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_pcm_experiment as pcm
from scripts import run_simulation_pilot as pilot
from scripts.experiment_utils.artifacts import atomic_write_json
from scripts.pilot_utils import summarize_pilot_cells


def generate_report(
    *, results_dir=pilot.OUTPUT_DIR, output_dir=None,
    minimum_successful=4, final_replications=None, ibs_target=0.003, cure_mse_target=0.005,
):
    results_dir = pilot.validate_output_dir(results_dir)
    manifest_path = results_dir / "pilot_design.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing pilot design: {manifest_path}. Run the pilot first.")
    saved_design = json.loads(manifest_path.read_text())
    design = pilot.build_pilot_design(
        replications=saved_design["replications"], seed=saved_design["master_seed"],
    )
    if saved_design != design:
        raise ValueError("The saved pilot design differs from the current simulation settings or software. Use the matching environment and source version.")
    frames = {}
    for cell in design["cells"]:
        frame = pilot.load_cell_result(results_dir, cell, design["study_id"])
        if frame is not None:
            frames[cell["task_id"]] = frame
    cells, report = summarize_pilot_cells(
        design["cells"], frames,
        planned_replications=design["replications"], minimum_successful=minimum_successful,
        final_replications=pcm.N_REPLICATIONS if final_replications is None else final_replications,
        ibs_target=ibs_target, cure_mse_target=cure_mse_target,
    )
    report.update(
        study_id=design["study_id"], pilot_master_seed=design["master_seed"],
        final_master_seed=design["final_master_seed"],
        pilot_results_dir=str(results_dir),
        allocation_rule="ceil((maximum within-cell sample SD / target MCSE)**2)",
        allocation_metrics=["test_ibs", "cure_mse"],
        sd_degrees_of_freedom=1,
    )
    destination = pilot.validate_output_dir(output_dir or results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "pilot_cell_summary.csv"
    json_path = destination / "pilot_precision_report.json"
    text_path = destination / "pilot_precision_report.txt"
    lines = [
        "Primary simulation precision pilot",
        f"Pilot design: {report['n_expected_cells']} cells, {design['replications']} planned replicates per cell",
        f"Pilot master seed: {design['master_seed']} (final experiment: {design['final_master_seed']})",
        f"Complete cells: {report['n_complete_cells']}/{report['n_expected_cells']}",
        f"Final replication count checked: {report['final_replications']}",
        f"Decision: {report['decision']}", "",
    ]
    if not report["complete"]:
        lines.extend(["Metric maxima and projections below are provisional until every planned cell is complete.", ""])
    for name, stats in report["metrics"].items():
        lines.append(
            f"{name}: maximum sample SD={stats['max_sd']}; "
            f"projected MCSE={stats['projected_mcse']}; "
            f"target={stats['target_mcse']}; required replications={stats['required_replications']}"
        )
    lines.extend(["", f"Overall required replications: {report['required_replications']}"])
    if report["problems"]:
        lines.extend(["", "Problems:", *[f"- {problem}" for problem in report["problems"]]])
    lines.extend([
        "", "Pilot observations are not pooled with final estimates.",
        "The final experiment count is not changed automatically.",
        "These measurements are recomputed with the recorded current implementation; historical manuscript values are not imposed.",
    ])
    pilot._write_csv(csv_path, cells)
    atomic_write_json(json_path, report)
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReports: {csv_path}, {json_path}, {text_path}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=pilot.OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, help="Default: the pilot results directory.")
    parser.add_argument("--minimum-successful", type=int, default=4)
    parser.add_argument("--final-replications", type=int, default=None, help="Default: the current final PCM experiment count.")
    parser.add_argument("--ibs-target", type=float, default=0.003)
    parser.add_argument("--cure-mse-target", type=float, default=0.005)
    args = parser.parse_args(argv)
    try:
        report = generate_report(**vars(args))
        if not report["ready_for_final"]:
            print("Pilot does not validate the configured final replication count; inspect the saved precision report before final estimation.", file=sys.stderr)
            return 2
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
