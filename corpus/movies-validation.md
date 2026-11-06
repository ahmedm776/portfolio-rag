# Movies data validation

DSAI202 coursework (Assignment 1) from the archive of Ahmed Khalifa, with an
AI-assisted post-coursework validation script. Ahmed's exact personal
contribution to the original assignment is unconfirmed. The original assignment
carried course instructions prohibiting AI-generated submission work; this
improved portfolio script is post-coursework work, not an assignment submission
and not claimed as Ahmed's unaided original code. The original is preserved unchanged.

## Verification scope

Tested 2026-11-04 on Python 3.10. Public mirror input: 9,999 rows, 9 columns.
Mirror: https://joannabieri.com/mathdatascience/data/movies.csv
Candidate dataset source: https://www.kaggle.com/datasets/bharatnatrayn/movies-dataset-for-feature-extracion-prediction
The public project provenance records byte-verified SHA-256 identity between the mirror and the original coursework CSV; findings apply to that dataset. Dataset rights were not established; no CSV, row-level outputs or full
profiling HTML is redistributed here. The generated profile was visually inspected.

## Conservative portfolio default

- Parse start year while retaining YEAR_RAW unchanged and missing years as missing.
- Use actual Cerberus validators and exact genre tokens; GX suite/checkpoint execute.
- Preserve all statistical extremes, flagged by IQR_FLAG, not removed as erroneous.
- No median imputation or scaling in default output. Raw snapshot preserved at runtime.
- Explicitly separate 80 missing-genre exclusions and 429 exact duplicates removed;
  sequential disjoint accounting: 9,999-80-429=9,490 default rows.
- Default keeps 579 missing years. 1,991 IQR-flagged rows stay in the default output.
- Optional --pedagogical branch mirrors coursework-style median imputation/scaling
  and IQR exclusion, explicitly labeled PEDAGOGICAL. It excludes 2,523 rows AFTER
  imputation, leaving 6,967. Different IQR counts arise because imputation changes
  quartiles; neither is a claim that extremes are errors. Missing years imputed in
  this branch are NOT recovered historical facts. YEAR_RAW remains unchanged.
- Inferred pandera schema describes observed types, not business validity.

## Run

python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python portfolio_movies.py /path/to/movies.csv --output outputs --profile
.venv/bin/python test_movies.py

Tests expect the verified-output folder from the reported run; for a fresh run use
--output verified-output. Test counts 136 and 9,999-row conservation are fixture-specific.
The profile is minimal=True, not the original full profiling configuration.
No learned models or predictive results are claimed. Original install/profiling cells
were excluded from the baseline reproduction; all other original code ran. Improved
script INCLUDING profiling ran successfully; 4 regression tests passed.

## Observed result

GX: FAIL, 4 of 12 expectations successful; 8 fail. This is a data-quality finding, not
an execution failure or a cleaned-data certification. Cerberus selected 136 rows.
See run_report.json and validation_summary.json for exact counts and provenance.
Start-year extraction found 9,251 source years before imputation. All final YEAR values
are nonnull, but missing years are explicitly median-imputed, not recovered facts.
The IQR-excluded rows and missing-genre rows are preserved when the script runs.
