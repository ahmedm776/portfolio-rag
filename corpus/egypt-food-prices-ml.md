# Egypt Food Price Spike Early Warning

End-to-end machine learning project: a classifier that estimates the
probability that a food commodity's average monthly price in Egypt will
rise more than 2% in the next month, trained on WFP food price data and
served through a FastAPI web app.

![App screenshot](figures/app_screenshot.png)

## Why this task

A first attempt at plain next-month price *regression* on this data found
that a persistence baseline ("next month = this month") is essentially
impossible to beat - Egyptian monthly food prices are smooth with step
jumps. Rather than ship a model that loses to its baseline, the project
reframes the problem as the question machine learning actually answers
better here: **is a price jump coming?** The classifier beats both the
majority-class and previous-month baselines on a strict time holdout.

## Results

**Protocol.** Chronological windows with every fitting date strictly before
every served date: train < Oct 2024, isotonic calibration on Oct 2024 -
Sep 2025, retrospective evaluation on Oct 2025 - Sep 2026. The pipeline
(model, features, 2% spike definition, 0.5 threshold) was frozen before the
evaluation run. This is a retrospective development evaluation: the window
was reproduced and reused across versions, so it is not an untouched
one-shot final test.

**Retrospective evaluation (Oct 2025 - Sep 2026, 215 series-months):**

| Metric | Model | Baseline |
|---|---|---|
| ROC AUC | **0.759** (all 215) | 0.500 (random). Paired on the 192 series-months with a prior month: model 0.750 vs 0.625 (previous month spiked) |
| Brier (raw / calibrated) | 0.141 / **0.138** | - |
| Accuracy (raw threshold 0.5) | 0.795 | 0.800 (majority) / 0.805 (previous month spiked) |
| Accuracy (calibrated threshold 0.5) | 0.800 | same baselines |
| F1 (spike class, raw threshold 0.5) | 0.353 | - |

Reading: the model carries real ranking signal - paired on the 192 scored
months with a prior month, model AUC 0.750 vs 0.625 for a prev-month-spike
rule - and calibration measurably improves its scores. At a 0.5 threshold its accuracy does NOT
beat trivial baselines on this window - reported as-is; the app presents a
calibrated risk score with baselines shown, not an accuracy claim.

**v4 model change (2026-11-04).** The classifier is now
HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03,
l2_regularization=1.0, max_leaf_nodes=15), replacing the v3
(max_iter=600) model. The variant was selected on rolling origins
2016-2022 ONLY (mean AUC .637 -> .664, min AUC .511 -> .529, mean
calibrated Brier .2391 -> .2218); the 2024-10 origin above was excluded
from selection. All origins are development-exposed data, not a holdout.
v4 leads v3 on 3 of 4 selection origins and trails on 2016; on no origin
does it beat the previous-month heuristic on accuracy. Full tables,
protocol freeze and negative findings: `ROBUSTNESS.md`,
`PROTOCOL_FREEZE.md`, `rolling_origin_eval.json`, `baseline_compare.json`.

**Honesty boundary.** The entire dataset was exposed during development
(task framing, model class, features and hyperparameters were chosen after
exploratory scoring of several windows). These numbers are a RETROSPECTIVE
evaluation of a frozen pipeline - there is no independent final test on
this dataset, and none is claimed. Full numbers and the protocol in
`models/metrics.json`; curves in `figures/roc_pr_curves.png` and
`figures/calibration_curve.png`.

**Late-issued October 2026 ledger.** A month-ahead spike-risk ledger for
October 2026 was frozen on 2026-11-05 - issued late, after the month it
covers, so it is a backfill (late-issued) evaluation ledger, not a
prospective forecast - and will be scored when WFP publishes October
actuals. The frozen scoring rule is in `prospective/score_ledger.py`.

## How leakage is prevented

- Features for a target month use only observations strictly before it.
- Lags are calendar-exact: `price_lag_k` is the price exactly k months
  earlier and `NaN` when that month was not observed (the Cairo series has
  gaps) - never a row-shift that smuggles future information across a gap.
- Chronological windows (train < 2024-10, calibrate 2024-10..2025-09,
  evaluate >= 2025-10), identical cutoffs for every series.
- The calibrator is fit only on the calibration window, on a model trained
  only on the training window - no future-to-past calibration.
- As-of serving rule: the app refuses to score months inside the training
  or calibration windows (before 2025-10) because those scores would be
  in-sample; every served score has all fit/calibration dates strictly
  before the scored month (asserted by tests).
- `tests/test_features.py` proves the no-leakage property: changing a
  month's price leaves that same month's features unchanged.
- `tests/test_split.py` proves the chronological windows are disjoint and
  global, that the artifact's train/calibration max dates strictly precede
  every servable date, that app prediction rows align with the exact
  preceding calendar months' real prices, and that in-sample history
  checks are refused.

## Project structure

```
src/features.py   calendar-exact lag + derived feature pipeline
src/train.py      trains HistGradientBoostingClassifier, writes model + metrics
src/evaluate.py   regenerates ROC/PR curves and holdout example plot
src/app.py        FastAPI app: web UI + JSON API
templates/        single-page UI
tests/            25 tests: leakage, artifact, API contract
data/             WFP/HDX source CSV (see data/DATA_SOURCE.md)
models/           model.joblib, metrics.json, holdout_predictions.csv
figures/          evaluation curves + app screenshot
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py        # reproduces models/ artifacts
python src/evaluate.py     # reproduces figures/
pytest tests/ -q           # 25 tests
uvicorn app:app --app-dir src --port 8000
# open http://localhost:8000
```

The API: `GET /api/series` lists the 47 series with evaluation metrics;
`POST /api/predict` returns the calibrated spike-risk score for the month
after the series' last observation (Oct 2026), with the model's training
and calibration date cutoffs attached - a retrospective as-of demo, not a
live forecast or verified probability; `POST /api/history-check` scores
the most recent scoreable month (>= Oct 2025, refusing in-sample months)
as-of the previous one and shows the actual outcome.
The UI supports deep links, e.g.
`/?market=Cairo%20(national%20average)&commodity=Rice&auto=1`.

## Honest limitations

- This is a screening signal, not a price forecast: it estimates jump
  risk, not the future price level.
- Only two price series exist in the source data (national averages);
  the model does not predict governorate-level prices.
- Macro shocks (currency devaluations) drive many spikes and are only
  indirectly visible through price momentum features.

## License

Code: MIT (see LICENSE). Data: WFP via HDX, CC BY-IGO - see
data/DATA_SOURCE.md for attribution.
