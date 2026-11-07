# Objective 1: shared K-means inputs

This bundle contains the complete file dependencies of `notebook/2_topic_discovery.ipynb`.
The user approved sharing these inputs and the one final pipeline on 2026-11-07.
The analytical sample has 10,000 r/jobs comments from January-October 2026, with
1,000 sampled comments per month. October ends at 2026-10-28 07:10:54 UTC and is partial.
The fitted topic model assigns 9,770 comments after excluding 106 moderation notices
and 124 zero-feature comments.

## Run from a fresh checkout

1. Use Python 3.13 and the isolated analytical dependencies in `requirements-objective1.txt`.
2. Open `notebook/2_topic_discovery.ipynb` from the repository root or `notebook/`.
3. Keep `RUN_TOPIC_MODEL=True` and `REUSE_SAVED_MODEL=True`, then run from the top.
   The default `RUN_DIR` points to this bundle's `prepared/` directory. No archive
   connection or model fitting is needed to reproduce the named analyses and figures.
4. To fit a new model, set `REUSE_SAVED_MODEL=False`. This uses the same cleaned corpus
   and fits TF-IDF, PCA and K-means. Follow the repository's approval rule before a
   full-corpus run taking more than a few minutes. New fits do not inherit old topic names.

The frozen pipeline was fitted with scikit-learn 1.7.0. The dedicated requirements
match its analytical stack. TensorFlow and GloVe are not required for this notebook.
The PCA50 baseline is read from the hash-verified comparison table; its historical
model is not included. The dimension figures use the recorded experiment tables,
so a 400-component PCA fit is not started automatically.

## Data and model distinctions

`raw_candidates.csv` is the original candidate collection. `eligible_pool.csv`,
`excluded_rows.csv` and `cleaning_audit.csv` document cleaning before sampling.
`comments_clean.csv` includes both original `text` and cleaned `text_topic`, as well
as timestamps and sampling strata. It is the direct modelling input. All six files
named in `sampling_manifest.json` are included because the notebook verifies them.

The fitted pipeline and its diagnostic CSV files are reference artifacts, not raw
observations. The notebook verifies their exact hashes before reuse. Only one final
pipeline is included. The five existing shared files in the parent directory remain
available for LDA and integration; no other notebook's data paths are changed.

## Complete payload inventory

| File | Rows | Role | Required for |
|---|---:|---|---|
| `prepared/raw_candidates.csv` | 20213 | Raw historical candidate comments, before cleaning | Acquisition audit and sampling integrity |
| `prepared/eligible_pool.csv` | 17766 | All candidates that passed cleaning, before stratified sampling | Sampling integrity |
| `prepared/excluded_rows.csv` | 2447 | Rejected candidates with reasons | Cleaning audit and sampling integrity |
| `prepared/cleaning_audit.csv` | 4 | Counts of retained and excluded candidate rows | Cleaning audit and sampling integrity |
| `prepared/comments_clean.csv` | 10000 | Selected corpus with original text, cleaned text, UTC timestamps and strata | Training and all analyses |
| `prepared/monthly_coverage.csv` | 10 | Monthly windows, sample quotas and partial-month flags | Training and all analyses |
| `prepared/sampling_manifest.json` | - | Sampling design, original counts and six input-file hashes | Training and all analyses |
| `prepared/data_quality_report.json` | - | Recorded acquisition and cleaning validation | Provenance |
| `acquisition_status.json` | - | Retrieval status and 200 time-stratum records | Sampling provenance |
| `collection_plan.json` | - | Frozen cutoff, sampling quotas and random seed | Sampling provenance |
| `prepared/topics_pca100_l2_20261029T105658930366Z/comment_topics.csv` | 10000 | Frozen assignments and exclusion status for every sampled comment | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/topic_profiles.csv` | 18 | Original cluster sizes and keyword profiles | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/monthly_topic_trends.csv` | 180 | Original monthly counts and topic shares | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/excluded_from_modelling.csv` | 230 | Comments excluded from topic modelling | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/zero_tfidf_comments.csv` | 124 | Comments with no retained TF-IDF features | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/representative_comments.csv` | 180 | Nearest-centroid and random examples for topic interpretation | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/k_selection.csv` | 6 | Candidate k diagnostics in the selected representation | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/model_comparison.csv` | 2 | Frozen PCA50 baseline and selected-model comparison | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/boundary_diagnostics.csv` | 18 | Cluster-level assignment ambiguity diagnostics | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/common_space_quality.csv` | 1 | Original TF-IDF cosine silhouette and term-overlap diagnostics | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/pca_variance.csv` | 100 | Selected 100-component PCA spectrum | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/pca_comparison.csv` | 2 | Fixed-50 and selected-dimension retained variance | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/initialization_stability.csv` | 2 | Alternative-seed assignment stability | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/tfidf_vocabulary.csv` | 10000 | Frozen vocabulary and IDF weights | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/monthly_assignment_coverage.csv` | 10 | Sampled versus assigned comments per month | Verified frozen-model reuse and integrity checks |
| `prepared/topics_pca100_l2_20261029T105658930366Z/run_parameters.json` | - | Exact frozen configuration and result-file hashes | Verified frozen-model reuse |
| `prepared/topics_pca100_l2_20261029T105658930366Z/topic_pipeline.joblib` | - | One final fitted TF-IDF/PCA/normalizer/K-means pipeline | Exact frozen assignments and named analyses |
| `prepared/topics_pca100_l2_20261029T105658930366Z/verification.json` | - | Recorded final-model validation | Provenance |
| `prepared/pca_dimension_comparison_20261029/comparison_manifest.json` | - | Input hash, dimensions, k grid and comparison settings | Dimension comparisons without refitting |
| `prepared/pca_dimension_comparison_20261029/pca_variance.csv` | 400 | Recorded 400-component spectrum for 50/100/200/400 comparisons | Dimension comparisons without refitting |
| `prepared/pca_dimension_comparison_20261029/comparison.csv` | 48 | All 48 dimension/normalization/k candidate diagnostics | Dimension comparisons without refitting |

`file_inventory.csv` records the same inventory with sizes, source paths and hashes.
`bundle_manifest.json` records every payload hash and the inventory hash. CSV and JSON
bytes are protected from automatic Git line-ending conversion through `.gitattributes`.
The coverage table restores the original ISO month and boolean formatting verified
against the existing sampling-manifest hash; no sampling values are changed.

## Interpretation and storage

The monthly quotas describe sample composition, not total subreddit volume. The
archive was sampled through bounded time-stratified candidate batches, not a census.
Names are assistant suggestions pending independent human review. Topic IDs apply
only to this verified model. Sensitivity checks hold assignments fixed and do not
establish population significance or causal labour-market effects.

Run outputs remain ignored by Git. Source-cache checkpoints, intermediate models,
the large comparison model bundle and generated figures are not included. These are
not required for offline training on the frozen corpus or for the existing analyses.
The acquisition records provide provenance, not the complete cache needed to resume
the original network collector.
