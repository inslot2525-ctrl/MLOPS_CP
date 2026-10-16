# Phase 2 evidence – dataset acquisition and validation

## Data source
The team supplied no Taiwan file (two zips given were German Credit, not used). The file was fetched from the public UCI repository on 2026-10-16:
`https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip` -> `default of credit card clients.xls` (UCI dataset 350, 2-row header). Converted with `scripts/prepare_raw.py` to `data/raw/credit_default.csv` (MD5 `df1b68b78adcea82ba83c8bf6a21757c`), tracked with `dvc add`.

## Commands and results
| Command | Result |
| --- | --- |
| `python scripts/prepare_raw.py <xls>` | wrote 30000 rows x 25 cols |
| `dvc add data/raw/credit_default.csv` | OK |
| `dvc repro` (prepare, split) | OK; `dvc status`: "Data and pipelines are up to date." |
| `ruff check .` | All checks passed |
| `pytest tests` | **39 passed, 2 skipped** (the 2 skipped are Postgres schema tests; Docker daemon off) |

## `scripts/profile_data.py` output (real data)
```
rows=30000 cols=25 validation_passed=True
errors: [] warnings: []
default_rate=0.2212
AGE min/max: 21 79
SEX counts: {1: 11888, 2: 18112}
EDUCATION counts: {0: 14, 1: 10585, 2: 14030, 3: 4917, 4: 123, 5: 280, 6: 51}
MARRIAGE counts: {0: 54, 1: 13659, 2: 15964, 3: 323}
PAY_0 counts: {-2: 2759, -1: 5686, 0: 14737, 1: 3688, 2: 2667, 3: 322, 4: 76, 5: 26, 6: 11, 7: 9, 8: 19}
age_band sizes: {'30-39': 11238, '18-29': 9618, '40+': 9144}
SEX x age_band:        n  default_rate
1 18-29             3281  0.2405
1 30-39             4565  0.2300
1 40+               4042  0.2558
2 18-29             6337  0.2222
2 30-39             6673  0.1837
2 40+               5102  0.2213
```
The profile agrees with the public contract in D.2 (30,000 rows, 22.12 % default, `PAY_0` with no `PAY_1`, EDUCATION {0,5,6} and MARRIAGE 0 present). No contradiction, so no stop condition. All six SEX x age_band cells are far above the 100-row eligibility minimum on the full data.

Split outputs: train 18000, eval 4500, test 3000, stream 4500 (9 batches of 500); train `f_ref` (flagged-cell fraction) = 0.0374.

## Exit criteria
- [x] Tests pass; profile saved here with age-band and SEX x band sizes.
- [ ] `dvc push` — **not done**: MinIO is not running (Docker daemon off). Remote `minio` (s3://dvc @ 127.0.0.1:9000) is configured but unused. Credentials must come from `AWS_*` env vars when pushing.

## Deviations / interpretations (see decision_log.md)
- `reference_stats.json` is produced by the `split` stage, not `prepare`, because it is computed from the train partition which only exists after the split.
- The outlier `f_ref` is the fraction of flagged *cells* (13 continuous columns x train rows); the spec did not say cell vs row.
- Great Expectations 1.24.0 and DVC 3.67.1 were used; suites are built from `configs/data.yaml` by `validate.py`, and `pipelines/expectations/*.json` are exports checked for sync by a test.
- Source is the public UCI file, not a team-supplied copy; N-02 (equivalence with Stage 1 code) remains unverified.
- Only `validate.py` imports Great Expectations; `ingest.py` uses a `BatchStore` protocol with a Postgres implementation (untested against a live DB) and an in-memory store used by the unit tests.
