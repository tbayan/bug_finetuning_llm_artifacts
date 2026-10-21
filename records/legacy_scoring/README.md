# Legacy scoring routine (not used for any number in the paper)

The master's thesis this work grew out of, and the evaluation screen of the CyberTriage
version it used, reported LLM scores that were computed by the routine in
`llm_metrics.py` (function `metrics_from_predictions_csv`). The routine appends synthetic
*correct* predictions, 25% as many as the real ones (500 on top of 2,000), before
computing accuracy, precision, recall and F1. `server_original.py` is the back end as it
was in the source repository; its function `_llm_rows_with_inflation` applied the routine
to every LLM row of the evaluation screen and multiplied the logged run time by 1.25.
`04-batch-evaluation-llms_padded.png` is the screenshot of that screen ("2500" samples,
fine-tuned Llama 3 at 62% F1, 3,274 s).

Section 1 of the paper discloses this. All numbers in the paper are computed from the
real per-report predictions in `results/*_predictions.csv` by
`analysis/rebuild_results.py`. The copy of `api/server.py` in this repository no
longer calls the routine and shows the logged metrics unchanged.

| Model | Padded value (thesis, old screen) | Real value (2,000 reports) |
|---|---|---|
| Llama 3 8B fine-tuned, macro-F1 | 0.62 | 0.526 |
| Qwen 2.5 7B fine-tuned, macro-F1 | 0.60 | 0.485 |
| Llama 3 8B zero-shot, macro-F1 | 0.48 | 0.305 |
| Qwen 2.5 7B zero-shot, macro-F1 | 0.49 | 0.321 |

These files are kept only as a record of how the earlier figures were produced.
