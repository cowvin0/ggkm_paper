"""Storage-only controls for the confirmatory study; estimands are unchanged."""

from contextlib import contextmanager
import gzip
import io
import json
import math
import os
from pathlib import Path

import numpy as np


def compact(design):
    return design["confirmatory_protocol"].get("storage", {}).get("format") == "compact-v1"


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def atomic_text(path):
    """Write plain/gzipped text atomically and remove incomplete temporary files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as raw:
            if path.suffix == ".gz":
                with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                        yield handle
            else:
                # detach keeps the binary stream open for fsync below.
                handle = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                try:
                    yield handle
                    handle.flush()
                finally:
                    handle.detach()
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    with atomic_text(path) as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False, separators=(",", ":"))


def summarize_traces(value):
    """Retain all starts/statuses; replace iteration-sized traces with summaries."""
    if isinstance(value, list):
        return [summarize_traces(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "loglik_history":
            finite = np.asarray([number for number in item if number is not None], dtype=float)
            result["loglik_history_summary"] = {
                "count": len(item), "finite": len(finite),
                "first": float(finite[0]) if len(finite) else None,
                "last": float(finite[-1]) if len(finite) else None,
                "minimum": float(finite.min()) if len(finite) else None,
                "maximum": float(finite.max()) if len(finite) else None,
                "decreases": int(np.count_nonzero(np.diff(finite) < -1e-8)),
            }
        elif key == "optimizer_results":
            stages = {}
            for row in item:
                stage = row["stage"]
                entry = stages.setdefault(stage, {"calls": 0, "failed": 0, "last_status": None,
                                                  "last_objective": None, "last_failure": None})
                entry["calls"] += 1
                entry["failed"] += not row.get("success", False)
                entry["last_status"] = row.get("status")
                entry["last_objective"] = row.get("objective")
                if not row.get("success", False):
                    entry["last_failure"] = row
            result["optimizer_summary"] = stages
        else:
            result[key] = summarize_traces(item)
    return result


def diagnostic_payload(payload, design):
    if compact(design) and design["stage"] != "audit":
        result = summarize_traces(payload)
        # Inner folds need selection/failure evidence, rather than a second
        # complete multistart journal. Keep the outer refit's starts in detail.
        for fit in result.get("fits", [])[:-1]:
            attempts = fit.pop("attempts", [])
            selected = fit.get("selected_attempt")
            fit["attempt_summary"] = attempt_summary(attempts)
            fit["selected_start"] = attempts[selected] if selected is not None else None
            for point in fit.get("profile", []):
                point["attempt_summary"] = attempt_summary(point.pop("attempts", []))
        return {**result, "trace_storage": "summarized-v1"}
    return payload


def attempt_summary(attempts):
    failures = [row for row in attempts if row.get("status") != "ok"]
    return dict(attempted=len(attempts), failed=len(failures),
                converged=sum(bool(row.get("converged")) for row in attempts),
                first_failure=failures[0] if failures else None,
                last_failure=failures[-1] if failures else None)


def allocated_bytes(path):
    stat = Path(path).stat()
    return max(math.ceil(stat.st_size / 4096) * 4096, stat.st_blocks * 512) + 2048


def storage_projection(root, design, final_replications, config):
    """Conservative pilot-based planning, not a filesystem quota guarantee.

    Use the largest observed per-replication allocation in EACH case, including
    filesystem blocks and an inode allowance; reserve uncompressed space for
    reporting even though the largest replication-level export is compressed.
    """
    from scripts import confirmatory_robustness_utils as confirm

    root = Path(root)
    pilot = root / "pilot"
    final_bytes, diagnostic_files, checkpoint_files = 0, 0, 0
    row_bytes = 0
    for case in design["cases"]:
        case_id = case["task_id"]
        sizes = []
        for checkpoint in sorted((pilot / "checkpoints" / f"case_{case_id:03d}").glob("replication_*.json.gz")):
            saved = read_json(checkpoint)
            size = allocated_bytes(checkpoint)
            for row in saved["rows"]:
                if "diagnostic_path" in row:
                    size += allocated_bytes(pilot / row["diagnostic_path"])
            sizes.append(size)
        if len(sizes) != design["replications"]:
            raise ValueError(f"Storage planning needs all pilot checkpoints: case {case_id}")
        final_bytes += max(sizes) * final_replications
        checkpoint_files += final_replications
        diagnostic_files += len(confirm.variants(case)) * final_replications
        with (pilot / f"replications_case_{case_id:03d}.csv").open("rb") as handle:
            row_bytes = max(row_bytes, max(map(len, handle), default=0))
    # Raw case CSVs plus fixed-dispersion export, metadata, and report headroom.
    final_bytes += diagnostic_files * max(row_bytes, 512) * 2
    paired_rows = sum(24 * (2 if case["n"] == 600 else 1) * final_replications
                      for case in design["cases"] if case["study_part"] == "primary")
    final_bytes += paired_rows * 512
    # Exclude this projection's own outputs so repeating a pilot summary does
    # not change its signed allocation and invalidate existing final checkpoints.
    excluded = {"pilot/storage_plan.json", "pilot/allocation.json"}
    existing = sum(allocated_bytes(path) for path in root.rglob("*")
                   if path.is_file() and path.relative_to(root).parts[0] != "final"
                   and path.relative_to(root).as_posix() not in excluded and path.suffix != ".tmp")
    projected = existing + int(config["storage"]["projection_factor"] * final_bytes)
    budget = config["storage"]["budget_gib"] * 1024 ** 3
    return dict(format="compact-v1", projected_bytes=projected,
                projected_gib=projected / 1024 ** 3, budget_gib=config["storage"]["budget_gib"],
                within_budget=projected <= budget, final_replications=final_replications,
                projected_checkpoint_files=checkpoint_files, projected_diagnostic_files=diagnostic_files,
                projection_factor=config["storage"]["projection_factor"],
                note="Pilot-based estimate with block/inode allowance and reporting reserve; not a guarantee of user quota or inode availability. Other files in the account are outside this budget.")
