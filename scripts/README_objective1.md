# Objective 1: monthly r/jobs topics

Open `notebook/2_topic_discovery.ipynb`. It reuses `cleanup_text` and the existing
CSV reader. The new `functions/topic_data.py` keeps topic-corpus rules separate
from the toxicity test-set preparation; existing shared behaviour is unchanged.

The current notebook uses TF-IDF -> centered PCA (100 components) -> row L2
normalization -> K-means (18 clusters, 30 initializations). The default reporting
mode is `REUSE_SAVED_MODEL=True`: it loads the already reviewed local model and
verifies input, assignment and pipeline SHA-256 values before applying names.
No TF-IDF, PCA or K-means is fitted in this reporting mode. Archive/network
acquisition stays disabled; earlier result files are preserved.

## Named clusters and focused reporting

The notebook embeds 18 suggested English topic names, interpretation confidence,
semantic limitations and trend judgments in `TOPIC_REVIEW`. These are assistant
interpretations, explicitly pending human validation. A changed or missing saved
model stops reuse; disabling reuse trains a new model with unreviewed names and
does not transfer old names by cluster number.

The main analysis now shows the six user-specified cases **C2, C5, C8, C9, C11,
and C17**, replacing the earlier strict interpretability-only filter. The case
review table presents semantic evidence, original interpretation confidence,
comment counts and overall shares, project relevance, and trend robustness.
Ratings are not upgraded to include these cases. The selection covers rising,
falling, phase-dependent and stable discussion, and remains exploratory.

All 18 clusters remain in the named size chart, PCA projection colorbar and
exported tables. The monthly heatmap hides only the C7 row AFTER all shares are
computed; it retains each visible value and its original denominator. Its 17
visible rows therefore need not sum to 100% within a month. A common linear
color scale uses the visible maximum (about 13.3%, instead of about 33.4% with
C7) and cells show their actual percentages. There is no row standardization,
nonlinear color transformation or renormalization. C12 remains visible.

The six main panels also retain all assigned comments as their denominator;
they do not renormalize the selected topics to 100%. The additive
`functions/topic_reporting.py` computes comment shares, equal sampled-post shares,
largest-post contributions and leave-one-time-stratum-out ranges. Those ranges
are sensitivity diagnostics, not confidence intervals. January-March versus
July-September comparisons exclude partial October.

New `topics_named_pca100_l2_<timestamp>/` exports include `topic_name_review.csv`,
`trend_summary.csv`, `focus_trend_summary.csv`, `focus_case_review.csv`,
`focus_robustness.csv`, `heatmap_visible_shares.csv`, named assignments and
examples, the six-panel trend figure, and model reuse/provenance flags. Names are not marked
as human reviewed merely because they are displayed. Source assignments and
weights are unchanged. Figure text and review metadata use English with the
portable DejaVu Sans font. The GitHub copy of the notebook has no saved outputs.

Verification includes an end-to-end run with model-fit methods blocked, a test
rejecting changed model files before deserialization, unchanged heatmap values,
known post-weight/denominator examples and missing/partial-month behavior.

## Dimension and distance comparison (2026-10-29)

With the identical 9,770 comments and 10,000-feature vocabulary, 50, 100, 200 and
400 PCA components retain 8.71%, 14.41%, 22.87% and 34.80% variance. The initial
48 candidates compare each dimension with/without row normalization and k in
6, 8, 10, 11, 12, 15. Local results are in
`prepared/pca_dimension_comparison_20261029/`. Additional k=18/20 candidates use
30 initializations; their tables are in `pca_domain_comparison_20261029/extended_*`.

More retained variance alone did not improve topic separation. The selected
100-dimensional model balances retained information, mixed-cluster size and
example interpretation. k=18 retains a recognizable AI-related cluster; the
inspected k=20 alternative instead has a cluster driven by the generic word
`use`. Names remain unreviewed, and not every cluster is a substantive topic.
The maximum cluster is smaller than the original 54.73%, but all-cluster
semantic separation has NOT been established. Check the executed notebook's
model comparison, original-space cosine silhouette, ARI and random examples.

An additional 24-candidate experiment tests domain stop words with protected
phrases such as `job market` and `laid off`. Although some term profiles are
clearer, initialization stability deteriorates. This experiment is not the
default. `vectorize_comments` keeps its prior behavior unless protected phrases
are explicitly supplied. `topic_feature_text` must also be applied at inference
if this optional mode is ever used. No original text is rewritten.

Variance is measured BEFORE row normalization, relative to the same TF-IDF
matrix. Normalization changes the distance geometry; PCA-space silhouette and
inertia across representations are not direct quality rankings. The notebook
also compares old/new assignments using identical TF-IDF vectors and a common
1,500-comment cosine-distance subset. Absolute cosine silhouettes remain low.
Changing dimension, normalization, k and initialization count together means
the overall difference cannot be attributed to PCA dimension alone.

New results use a fresh `topics_pca100_l2_<timestamp>/` directory. The serialized
pipeline includes vectorizer, PCA, normalizer and K-means. Prediction must use
that sequence, after the existing topic cleaning. Exports preserve all 10,000
IDs, including 230 exclusions, and add relative centroid assignment margins.
These margins are ambiguity diagnostics, not probabilities. No low-margin
comments are dropped to make monthly curves look clearer.

The optional `reduce_topics_to_variance` 90% helper remains available but was
not executed on the real corpus. It needs dense covariance memory and separate
long-run authorization. The proposal specified PCA50; report the requested
dimension change and baseline comparison explicitly.

Reproduce the initial dimension comparison (takes minutes; follow AGENTS.md):

```powershell
python scripts/compare_topic_models.py --baseline dataset/reddit/topics_2026/time_stratified_20261028/prepared/topics_20261029T034634149868Z --output dataset/reddit/topics_2026/time_stratified_20261028/prepared/NEW_COMPARISON
```

References: [scikit-learn text clustering example](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html).
Its normalization discussion uses SVD; this project separately tests centered
PCA, so improvements cannot be assumed from that example.

Completed on 2026-10-28: 1,000 cleaned comments for each of January through
October 2026, 10,000 unique comment IDs in total. The frozen acquisition cutoff
is 2026-10-28 07:10:54 UTC (15:10:54 Singapore); October is partial. From 20,213
candidate rows, cleaning retained 17,766 eligible rows and excluded 2,447:
1,634 deleted/removed, 804 too short and 9 empty after cleaning. The final
10,000-row stratified sample is selected from those eligible candidates.

`prepared/data_quality_report.json` records the successful acquisition quota,
uniqueness, raw-text preservation, cleaning, hash and original data-cell checks.
New model runs have separate `run_parameters.json` and `verification.json` files.
The earlier transient September
failure is preserved in `acquisition_attempt_01.json`; the completed run resumed
from source checkpoints.

## Previous fixed-50 baseline (2026-10-29)

The additive `functions/topics.py` provides topic-only feature and diagnostic
helpers. Shared `cleanup_text` and the toxicity preprocessing are unchanged.
The initial configuration grouped about 99% of comments together while isolating
moderation templates. The documented topic configuration therefore excludes
explicit moderation notice phrase combinations, applies standard English stop
words plus listed conversational fillers, and retains `off` for `laid off`.
This changes modelling features only; the original corpus stays at 10,000 rows.

The fixed-50 baseline run excludes 106 moderation notices and 124 zero-vector comments,
leaving 9,770 rows with 10,000 TF-IDF features. True ARPACK PCA reduces these to
50 dimensions and retains about 8.71% of variance. k=2 through 12 are evaluated
using inertia, a fixed 1,500-row silhouette subset and cluster size diagnostics.
k=11 was its exploratory default (silhouette about 0.110); its extra-seed ARI is
about 0.63-0.66, so boundaries remain initialization-sensitive. Names default to
unreviewed identifiers, with original-space mean TF-IDF terms and example text
available for naming. Do not describe the clusters as definitive ground truth.

Each model run creates a new `prepared/topics_<timestamp>/` directory with
assignments for every input ID, exclusion reasons, vocabulary, PCA variance,
k diagnostics, initialization stability, profiles, representative comments,
monthly topic shares, figures and `topic_pipeline.joblib`. `run_parameters.json`
contains input/output hashes and exact settings. Monthly assignment coverage
reports the difference between all sampled comments and assigned comments.

If a spreadsheet program changes only coverage month labels or boolean casing,
the loader can restore metadata in memory only when reconstructed bytes match
the original manifest hash exactly. Changed numerical contents or changed
comment data still stop execution. No input files are overwritten.

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