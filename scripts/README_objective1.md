# Objective 1: monthly r/jobs topics

Open `notebook/2_topic_discovery.ipynb`. It reuses `cleanup_text` and the existing
CSV reader. The new `functions/topic_data.py` keeps topic-corpus rules separate
from the toxicity test-set preparation; existing shared behaviour is unchanged.

The current work is acquisition and cleaning only. Keep `RUN_TOPIC_MODEL=False`;
the optional modelling sections have not been run on this corpus.

Completed on 2026-10-28: 1,000 cleaned comments for each of January through
October 2026, 10,000 unique comment IDs in total. The frozen acquisition cutoff
is 2026-10-28 07:10:54 UTC (15:10:54 Singapore); October is partial. From 20,213
candidate rows, cleaning retained 17,766 eligible rows and excluded 2,447:
1,634 deleted/removed, 804 too short and 9 empty after cleaning. The final
10,000-row stratified sample is selected from those eligible candidates.

`prepared/data_quality_report.json` records the successful quota, uniqueness,
raw-text preservation, cleaning, hash and notebook-cell checks. Five notebook
data cells were executed in order, through data exploration; no feature
extraction, PCA or clustering was executed. The earlier transient September
failure is preserved in `acquisition_attempt_01.json`; the completed run resumed
from source checkpoints.

## Current collection: time-stratified candidates

`scripts/collect_monthly_topics.py` uses Arctic Shift's historical comment search.
Each UTC month is split into 20 equal-duration time strata. A deterministic
random anchor in each stratum defines a limited candidate batch, fetched in time
order; if necessary the collector continues within the stratum or wraps to its
start. After cleaning, exactly 50 eligible comments are sampled without
replacement from each stratum (1,000 per month, seed 42).

This is a time-stratified random-anchor candidate design, not a simple random
sample of every comment in the month. Inclusion probabilities depend on local
activity. No topic keywords or toxicity labels are used. The archive's full
monthly population is not enumerated; `source_window_complete=False` records
that limitation. Equal monthly quotas cannot estimate absolute subreddit volume.

The current run is `dataset/reddit/topics_2026/time_stratified_20261028/`; the
notebook points to its `prepared/` subdirectory. Each stratum has a local source
checkpoint, anchor, bounds, request intervals, candidate counts and bot flags.
Usernames are not stored. Comment permalinks are constructed from source IDs,
not independently visited. Missing source language labels remain unverified.

The user approved this longer rate-limited collection on 2026-10-28. To resume
an interrupted run, use the same command; its original cutoff and seed are kept:

```powershell
python -u scripts/collect_monthly_topics.py --work-dir dataset/reddit/topics_2026/time_stratified_20261028
```

An already completed `prepared/` directory is never overwritten. For a fresh
collection, choose a new work directory. The rest of this guide also describes
the older exhaustive-window collector and CSV import option.

## Data acquisition

The target is 1,000 eligible comments per UTC month, January 2026 through the
acquisition cutoff. October is partial as of 2026-10-28; November and December
do not exist yet. Sampling happens after cleaning, without replacement and with
seed 42. If fewer than 1,000 eligible comments are supplied, retain the available
rows and report a shortfall. Do not duplicate comments or use the balanced
100-comment toxicity sample to fill gaps.

The existing `collect_reddit.py collect` only fetches recent listing comments;
it cannot backfill historical months. The new bounded archive command queries
historical windows and splits full 100-record responses into smaller time
windows. It only samples a month after its archive window has been exhausted.
This establishes completeness relative to the archive response, not to Reddit.
Failed requests stop collection and preserve an acquisition report; unattempted
months are not zero-activity months. No retries are made on rate limits.

From the repository root, choose a fresh run directory:

```powershell
python scripts/prepare_topic_data.py archive --output-dir dataset/reddit/topics_2026/archive_run02
```

The default request budget is 100 requests / 90 seconds. A large monthly corpus
may exceed that budget and remain incomplete. Do not silently raise it into a
long run; the repository requires asking before runs taking more than a few
minutes. The current implementation preserves completed months, but does not
checkpoint partial recursive windows.

For a local historical CSV, with `comment_id,text,created_utc,subreddit` and Unix
seconds, use:

```powershell
python scripts/prepare_topic_data.py import-csv --input dataset/reddit/your_export.csv --output-dir dataset/reddit/topics_2026/import_run01
```

Optional input columns: `link_id,parent_id,permalink,collected_utc,is_bot,language`.
An `author` column, if supplied, is used only to filter AutoModerator and is not
saved. Export provenance and population coverage must be documented separately;
importing a CSV cannot establish that it is representative or exhaustive.

## Cleaning and outputs

- Preserve `text`; create `text_topic` using HTML entity decoding, lowercase and
  the shared `cleanup_text`.
- Remove invalid IDs/timestamps, non-jobs rows, out-of-window comments, missing
  bodies, deleted/removed text and known bots.
- Filter non-English rows only when source language metadata is available.
  Otherwise flag language as unverified and inspect a random sample manually.
- Default to at least 3 alphabetic words after cleaning; record exclusions.
- Deduplicate comment IDs. Fail on conflicting snapshots. Retain matching text
  from different IDs and flag it for review.
- Exclude zero TF-IDF rows later in the notebook and retain their IDs in exports.

Each local run includes `comments_clean.csv`, `raw_candidates.csv`,
`eligible_pool.csv`, `excluded_rows.csv`, `cleaning_audit.csv`,
`monthly_coverage.csv`, and `sampling_manifest.json`. Archive runs additionally
include `acquisition_status.json`. The manifest records parameters and file
hashes. Outputs stay under the ignored `dataset/reddit/topics_2026/` directory.
Existing run directories are never overwritten.

Equal monthly quotas support comparisons of sample topic shares. They do not
measure total subreddit discussion volume. A current partial month and missing
archive months must remain visible in any report.

## Verification

```powershell
python -m unittest discover -s tests -p test_topic_data.py
```

Notebook modelling is opt-in (`RUN_TOPIC_MODEL=True`). Examine the k-selection
plots, set `SELECTED_K`, then name topics from original-space mean TF-IDF terms
and representative comments. The notebook does not auto-select k or invent
topic names. No TensorFlow, GloVe download or saved-model overwrite is involved.

## Source check on 2026-10-28

Initial Arctic Shift probes returned HTTP 500 and the first bounded run stopped
on that error; its record remains in `archive_20261028/acquisition_status.json`.
A later check succeeded, including a January-2026 request. The resumed task now
uses the available Arctic Shift service for the time-stratified collection.
A PullPush probe had returned HTTP 429 and was not retried. Hugging Face's
advertised archive did not expose the expected 2026 directory, and its server
filter timed out; large all-Reddit torrent downloads were not needed.

Source documentation:

- https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md
- https://support.reddithelp.com/hc/en-us/articles/49381918834964-Reddit-for-Researchers-Program

The official research program documents a six-month data delay, so it cannot be
assumed to provide every recent 2026 month either. Public archive availability
and archive coverage are separate questions from the project's data-use terms.
