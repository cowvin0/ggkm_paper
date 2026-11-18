"""Compose the paper's IBS/concordance figures without refitting any model.

Run from the repository root with the experiment Python environment:
    python figures/compose_paper_score_panels.py

Existing vector plots and saved standalone scores are the inputs. Clinical
calibration/error panels, legends, model choices, and plotted split scores are
preserved. Comparisons use one shared legend and one shared x-axis title.
Only the clinical concordance panel is drawn anew, using the same
plotting helper. Requires pdflatex and the Poppler PDF utilities.
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts import melanoma_auc_plot as plot

DIRECTORY = ROOT / "figures/review"


def pdf_size(path):
    xml = subprocess.check_output(["pdftotext", "-bbox", str(path), "-"])
    pages = [node for node in ElementTree.fromstring(xml).iter() if node.tag.endswith("page")]
    if len(pages) != 1:
        raise ValueError(f"Expected one page: {path}")
    return tuple(float(pages[0].attrib[key]) for key in ("width", "height"))


def word_top(path, word):
    """Locate a shared label in the original, text-searchable vector plot."""
    xml = subprocess.check_output(["pdftotext", "-bbox", str(path), "-"])
    matches = [float(node.attrib["yMin"]) for node in ElementTree.fromstring(xml).iter()
               if node.tag.endswith("word") and node.text == word]
    if len(matches) != 1:
        raise ValueError(f"Expected one {word!r} label in {path}")
    return matches[0]


def compose(destination, size, graphics, *, letters=()):
    """Place vector pages using PDF-point coordinates and optional clipping."""
    width, height = size
    commands = []
    for path, x, y, w, h, trim in graphics:
        options = f"width={w}bp,height={h}bp"
        if trim:
            options += ",clip,trim=" + " ".join(f"{value}bp" for value in trim)
        commands.append(rf"\put({x},{y}){{\includegraphics[{options}]{{\detokenize{{{path}}}}}}}")
    for text, x, y in letters:
        commands.append(rf"\put({x},{y}){{\sffamily\bfseries\fontsize{{12}}{{14}}\selectfont {text}}}")
    source = rf"""\documentclass{{article}}
\usepackage[paperwidth={width}bp,paperheight={height}bp,margin=0bp]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\topskip}}{{0pt}}
\begin{{document}}
\setlength{{\unitlength}}{{1bp}}
\noindent\begin{{picture}}({width},{height})
{chr(10).join(commands)}
\end{{picture}}
\end{{document}}
"""
    with tempfile.TemporaryDirectory(prefix="ggkm-score-panels-") as temporary:
        directory = Path(temporary)
        tex = directory / "panels.tex"
        tex.write_text(source)
        subprocess.run(["pdflatex", "-halt-on-error", "-interaction=batchmode",
                        f"-output-directory={directory}", str(tex)], check=True, capture_output=True)
        # Re-emit visible vector content so clipped-away metric text is absent
        # from extraction as well as from the displayed page.
        cleaned = directory / "visible.pdf"
        subprocess.run(["pdftocairo", "-pdf", str(directory / "panels.pdf"), str(cleaned)], check=True)
        pdf_size(cleaned)
        text = subprocess.check_output(["pdftotext", str(cleaned), "-"], text=True)
        if re.search(r"\bAUC", text, re.I):
            raise ValueError(f"Unexpected cure-AUC content in {destination}")
        shutil.copyfile(cleaned, destination)
    print(destination.relative_to(ROOT))


def concordance_panel(prefix, destination):
    scores = pd.read_csv(DIRECTORY / f"{prefix}_standalone_metric_values.csv")
    scores = scores.loc[scores.comparison.eq("models") & scores.metric.eq("uno_c_index")]
    colors = plot.plt.rcParams["axes.prop_cycle"].by_key()["color"]
    specs, arrays = [], []
    labels = {"mixture_cure": "Mixture\ncure", "promotion_time": "Promotion\ntime",
              "spline_poisson": "Spline\nP", "spline_bn": "Spline\nNB"}
    for index, ((model, kernel), rows) in enumerate(scores.groupby(["model_name", "kernel"], sort=False)):
        label = labels.get(model)
        if label is None:
            short = "P" if model == "poisson" else "NB"
            label = f"GG-KM {short}\n{plot.KERNEL_LABELS[kernel]}"
        specs.append(dict(box_label=label, color=colors[index],
                          hatch="//" if model in {"bn", "spline_bn"} else None))
        arrays.append(rows.sort_values("split").value.to_numpy())
    if len(specs) != 6 or any(not np.isfinite(values).all() for values in arrays):
        raise ValueError(f"Incomplete saved concordance scores: {prefix}")
    fig, axis = plot.plt.subplots(figsize=(5.4, 5.6))
    lower = max(0., min(.48, float(np.min(np.concatenate(arrays))) - .03))
    plot.draw_score_panel(axis, arrays, specs,
                          title="C  Paired repeated-holdout survival discrimination",
                          ylabel="Uno's C-index", reference=.5, ylim=(lower, 1.))
    fig.subplots_adjust(bottom=.25, top=.92, left=.15, right=.985)
    fig.savefig(destination, bbox_inches="tight")
    plot.plt.close(fig)


def compose_kernel_comparison(prefix):
    left = DIRECTORY / f"{prefix}_ggkm_kernel_comparison_ibs.pdf"
    right = DIRECTORY / f"{prefix}_ggkm_kernel_comparison_uno_c_index.pdf"
    lw, lh = pdf_size(left)
    rw, rh = pdf_size(right)
    left_strip = lh - word_top(left, "Poisson") + 5.
    right_strip = rh - word_top(right, "Poisson") + 5.
    compose(DIRECTORY / f"{prefix}_ggkm_kernel_comparison_ibs_uno_c_index.pdf",
            (lw + rw, max(lh, rh)), [
                (left, 0., left_strip, lw, lh - left_strip, (0., left_strip, 0., 0.)),
                (right, lw, right_strip, rw, rh - right_strip, (0., right_strip, 0., 0.)),
                # Reuse one original legend, centered beneath both panels.
                (left, rw / 2, 0., lw, left_strip, (0., 0., 0., lh - left_strip)),
            ], letters=[("A", 8., lh - 15.), ("B", lw + 8., rh - 15.)])


def compose_simulation_comparison(stem):
    upper, lower = (DIRECTORY / f"{stem}_{suffix}.pdf" for suffix in ("ibs", "uno_c_index"))
    uw, uh = pdf_size(upper)
    lw, lh = pdf_size(lower)
    width = max(uw, lw)
    # Keep every tick label and plotted mark; omit only the upper row's
    # shared axis title and legend. The lower row supplies both for the figure.
    upper_height = word_top(upper, "Generating") - 6.
    gap = 10.
    compose(DIRECTORY / f"{stem}_ibs_uno_c_index.pdf", (width, upper_height + lh + gap), [
        (upper, (width - uw) / 2, lh + gap, uw, upper_height, (0., uh - upper_height, 0., 0.)),
        (lower, (width - lw) / 2, 0., lw, lh, None),
    ])


def main():
    with tempfile.TemporaryDirectory(prefix="ggkm-concordance-") as temporary:
        for prefix in ("melanoma", "breast_cancer", "fosp_melanoma"):
            original = DIRECTORY / f"{prefix}_predictive_assessment_ibs_auc.pdf"
            width, height = pdf_size(original)
            split, legend = 840., 65.
            panel = Path(temporary) / f"{prefix}.pdf"
            concordance_panel(prefix, panel)
            compose(DIRECTORY / f"{prefix}_predictive_assessment_ibs_uno_c_index.pdf",
                    (width, height), [
                        (original, 0., legend, split, height - legend, (0., legend, width - split, 0.)),
                        (original, 0., 0., width, legend, (0., 0., 0., height - legend)),
                        (panel, split, 95., width - split, height - 95., None),
                    ])
            compose_kernel_comparison(prefix)
        for stem in ("simulation_rbf_paired_differences",
                     "highdim_rbf_vs_linear_spline_polynomial_paired_differences"):
            compose_simulation_comparison(stem)


if __name__ == "__main__":
    main()
