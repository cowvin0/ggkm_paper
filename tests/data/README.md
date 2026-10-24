# Paper model reference results

`legacy_models.json` is an unchanged copy of the original library regression
fixture, captured before the estimator API conversion. Its original source
hashes and dependency versions remain in the JSON. The paper's
`test_paper_other_models.py` uses its inputs and primal/spline configurations;
the mixture and promotion cases remain covered in the library's tests.

`primal_models_832dcf1.json` records the five primal and spline fits with
scikit-cure commit `832dcf1e5022171cb7b7df9996198c4752e00b64`. It uses the same
data, constructor parameters, three EM iterations, and prediction times as
`legacy_models.json`. The updated backend rescales incidence optimization and
retries failed generalized-gamma latency line searches, so the original fitted
numbers no longer describe this dependency version. The new reference includes
the backend commit, dependency versions, input checksum, and adapter checksums.
It supplies the numerical regression expectations in `test_paper_other_models.py`;
`test_primal_compatibility.py` independently checks the original-unit penalty and
its gradient. The spline design is still checked against the original fixture.

`primal_em_reference.json` is the unchanged reference captured before the
primal EM loops were extracted. Its successful fits, optimizer traces, and
failed-refit states are checked by `test_paper_em_primal.py`. See
`primal_em_reference.md` for the original capture details.

`paper_spline_move_reference.json` was captured immediately before these
estimators moved into `ggkm_paper`. Its spline features, estimator states,
predictions, and optimizer traces are checked bit-for-bit by
`test_paper_spline_models.py`. See `paper_spline_move_reference.md` for details.
