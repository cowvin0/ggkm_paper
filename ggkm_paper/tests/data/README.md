# Paper model reference results

`legacy_models.json` is an unchanged copy of the original library regression
fixture, captured before the estimator API conversion. Its original source
hashes and dependency versions remain in the JSON. The paper's
`test_paper_other_models.py` checks only its primal and spline cases; the
mixture and promotion cases remain covered in the library's tests.

`primal_em_reference.json` is the unchanged reference captured before the
primal EM loops were extracted. Its successful fits, optimizer traces, and
failed-refit states are checked by `test_paper_em_primal.py`. See
`primal_em_reference.md` for the original capture details.

`paper_spline_move_reference.json` was captured immediately before these
estimators moved into `ggkm_paper`. Its spline features, estimator states,
predictions, and optimizer traces are checked bit-for-bit by
`test_paper_spline_models.py`. See `paper_spline_move_reference.md` for details.
