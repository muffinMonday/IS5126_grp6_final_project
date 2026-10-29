# Datasets

Only explicitly approved data files are committed. Raw downloads and other generated
outputs remain local and ignored by Git (see `.gitignore`).

| File / folder | Source | In git? |
|---|---|---|
| `youtoxic_english_1000.csv` | [YouTube Toxicity Data](https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data) (label: `IsToxic`) | yes (small) |
| Jigsaw Toxic Comment Classification (Wikipedia talk pages) | [Kaggle competition](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) | no |
| Jigsaw Unintended Bias / Civil Comments (news comments) | [Kaggle competition](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) | no |
| GOSU.ai English Dota 2 chat | [Kaggle dataset](https://www.kaggle.com/datasets/danielfesalbon/gosu-ai-english-dota-chat) | no |
| `pooled/pooled_train.csv` | Part 3 cleaned training split, 19,187 rows | yes (explicitly approved) |
| `pooled/pooled_val.csv` | Part 3 fixed validation split, 4,797 rows | yes (explicitly approved) |
| `pooled/manifest.json` | Part 3 settings, source hashes and split hashes | yes (explicitly approved) |
| `reddit/test_100_ai_draft.csv` and `reddit/collection_manifest.json` | 100 r/jobs comments with AI preliminary labels and provenance | yes (explicitly approved) |
| Other r/jobs candidates, browser exports and human review CSVs | collected by the team | no |

Label mapping to toxic / non-toxic is described in the project proposal (`Project Proposal/`).
Put new downloads in this folder; files outside the approved list stay untracked automatically.

## Pooled training data

The user approved these three handoff files on 2026-10-29. Package 4 can use them
after pulling the repository, without downloading the four original sources or
rerunning Part 3. Each CSV contains `orig_id, source, text_raw, text, label`.
Read with `keep_default_na=False` and `dtype={'orig_id': str}`; use the already
cleaned `text` as input and binary `label` as target. Fit the tokenizer and tune
models only on `pooled_train.csv`; reserve `pooled_val.csv` for final evaluation.

`manifest.json` records the source mappings, seed, preparation version and exact
SHA-256 hashes of both CSVs. Git preserves their bytes across platforms so these
hashes can be checked before training. Any replacement must keep the two splits
and manifest consistent.

Other Part 3 outputs are not tracked:

| Output | Local files |
|---|---|
| Scores | `baseline_metrics.csv`, `loso_metrics.csv`, `transfer_comparison.csv` |
| Predictions | `baseline_predictions.csv`, `loso_predictions.csv` |
| Errors and removal audit | `baseline_errors.csv`, `error_counts.csv`, `removed_rows.csv` |
| Summaries | `data_summary.csv`, `text_length_summary.csv`, `results_summary.txt` |
| Figures | `figures/source_and_length.png`, `figures/baseline_confusion_matrices.png`, `figures/baseline_pr_curves.png` |

To regenerate the handoff and these results, use the original downloads below.

`notebook/3_pooled_data.ipynb` expects the Kaggle downloads in folders named after their Kaggle
dataset / competition; only the files listed are read:

```
dataset/
  youtoxic_english_1000.csv
  jigsaw-toxic-comment-classification-challenge/train.csv.zip      (read as zip)
  jigsaw-unintended-bias-in-toxicity-classification/train.csv
  gosu-ai-english-dota-chat/tagged-data.csv
  pooled/                                                         (created by notebook 3)
```

## Reddit collection and annotation

### Shared AI-labelled sample (2026-10-28)

The committed `dataset/reddit/test_100_ai_draft.csv` contains **100 actual r/jobs comments
from 58 posts**, selected from 287 deduplicated candidates read through the
user-verified Reddit browser. The comment creation dates range from 2022-10-17
to 2026-10-13. Collection took place on October 27–28, 2026 (Singapore time).
No synthetic comments were included. The user explicitly authorized sharing
this 100-row sample and its provenance in Git; other collection artifacts remain
excluded. Labels assess toxicity, not positive/negative sentiment.

- `test_100_ai_draft.csv`: 50 suggested non-toxic labels and 50 suggested toxic
  labels, with source URLs and individual rationales. **All labels are AI
  preliminary annotations**, `annotation_status=pending`, `annotator=AI:Codex`.
  This file is tracked.
- `collection_manifest.json`: tracked acquisition method, dates, selection
  limitations, counts and source/output SHA-256 hashes. References to local-only
  files document provenance; those files are not included in a fresh clone.
- `review_100.csv` (local only): the same 100 comments with blank labels/reviewer
  fields for independent human annotation. After review, use `finalize` on it.
- `browser_candidates.csv` (local only): 287 deduplicated candidates with blank labels.
- `browser_*.json` (local only): saved source records; `ai_annotations_100.json`
  records the selected IDs and AI rationales.

From a fresh clone, the shared CSV is ready for exploratory inference. Read
`text` as the model input and `is_toxic` as the **AI-suggested** binary label.
`comment_id` is the join key; `annotation_notes` contains the reasoning for each
suggestion. Use `permalink` to inspect a source. Apply the saved training
preprocessing/tokenizer without refitting. Any scores against these labels must
be reported as agreement with AI annotations, not human-grounded accuracy.

To make a local blank review sheet, use the committed CSV directly:

```bash
python scripts/collect_reddit.py import-csv --input dataset/reddit/test_100_ai_draft.csv --output dataset/reddit/review_100.csv
```

Choose a new output name if that file already exists. This resets labels and
reviewer fields; it does not modify the committed sample.

The browser records contain rendered comment text, not API raw Markdown. Case
and words are preserved; Markdown formatting and inline-link destinations are
not retained. Source links identify the comment, while parent IDs are blank for
search results where they were unavailable. Snapshot file modification time is
used as the approximate acquisition time during import.

This is a **keyword-enriched convenience sample**, with additional comments from
two discussion threads. It overrepresents personal insults and clusters some
non-toxic examples in the same threads. It cannot estimate Reddit toxicity rates
or represent all types of toxicity. Human review is still required for the
proposal's gold test set; do not report scores against AI labels as human-grounded
evaluation. Browser availability does not establish research-use permission.

The original collector can reproduce the conversion below using the local-only
source snapshots and annotation JSON. These inputs are not shipped in Git.
Use new output names because existing files are never overwritten:

```bash
python scripts/collect_reddit.py import-browser --input dataset/reddit/browser_candidates_20261027.json dataset/reddit/browser_search_full_20261027.json dataset/reddit/browser_thread_2_20261027.json dataset/reddit/browser_search_2_20261027.json --output dataset/reddit/browser_candidates_v2.csv
python scripts/collect_reddit.py ai-draft --input dataset/reddit/browser_candidates_v2.csv --annotations dataset/reddit/ai_annotations_100.json --output dataset/reddit/test_100_ai_draft_v2.csv
# After the team has filled review_100.csv:
python scripts/collect_reddit.py finalize --input dataset/reddit/review_100.csv
```

`import-browser` reads a JSON array with `comment_id,text,created_iso,subreddit,
link_id,permalink` and optional `parent_id`. Dates must include a timezone and
links must match the r/jobs post/comment IDs. It does not automate login, browser
verification or website requests. `ai-draft` only attaches supplied suggestions;
it does not call a model or infer labels from keywords. Its annotation JSON maps
each selected ID to `{"label": 0, "reason": "..."}` or label `1`, exactly 50 of each.
The finalizer rejects pending rows and `AI:` reviewer IDs. Human reviewers must
make their own decisions and supply their own reviewer IDs.

Work package 1 produces a **100-comment, manually labelled r/jobs test set:
50 non-toxic and 50 toxic**. It is reserved for final zero-shot evaluation of the
classifier trained on external datasets. Do not add it to training, fit a tokenizer
on it, choose hyperparameters from it, or use it to tune a decision threshold.

### 1. Choose an available data source

For this academic project, start with Reddit for Researchers (RFR). Reddit's current
[access guidance](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data)
identifies RFR as the authorized research route, with access through BigQuery
Analytics Hub. A developer OAuth app alone does not establish research permission.
If the team has specific permission covering Data API collection for this project,
the existing OAuth collector can be used. Otherwise use an export allowed by the
research access you obtain. Reddit's Data API requires OAuth and an identifying
User-Agent; an ordinary Reddit login alone is insufficient.
The collector uses the official OAuth endpoint, bounded pagination, timeouts and
limited retries. It respects rate-limit headers and stops on a long wait or denied
access. It does not fall back to anonymous scraping.

Current access guidance, checked 2026-10-27:

- [Reddit Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki)
  describes authentication, User-Agent and rate limits.
- [Reddit for Researchers](https://support.reddithelp.com/hc/en-us/articles/49381918834964-Reddit-for-Researchers-Program)
  describes research applications and BigQuery Analytics Hub access. Research
  access is a separate route, not a promise of OAuth credentials. Use the import
  route below only if your access permits an offline export.
- [OAuth technical reference](https://github.com/reddit-archive/reddit/wiki/OAuth2#application-only-oauth)
  documents application-only tokens. Availability still depends on current access
  approval; legacy app-registration instructions may be outdated.

To request research access, follow the application link in the RFR page above and
select the researcher role. The published requirements include an institutional
email, the project proposal/data needs, an ethics approval or exemption letter,
and an institutional sponsor (for example, your instructor). Approval is not
instant or guaranteed. RFR access is not a client ID/client secret pair.

For separately authorized OAuth access, the application settings at
<https://www.reddit.com/prefs/apps> show the client ID and (for a confidential app)
client secret. App creation/access may require approval; merely registering an
account or inventing a User-Agent does not grant API access. The collector exchanges
approved app credentials for a temporary token automatically. Do not create an
unrelated app or misdescribe the research purpose to obtain access.

For authorized OAuth access, set environment variables in your local terminal.
Replace placeholders locally; never put real secrets in code, chat or Git.

macOS/Linux:

```bash
export REDDIT_USER_AGENT='script:is5126-group6:v1.0 (by /u/YOUR_REDDIT_USERNAME)'
export REDDIT_CLIENT_ID='YOUR_APPROVED_CLIENT_ID'
export REDDIT_CLIENT_SECRET='YOUR_CLIENT_SECRET'
```

PowerShell:

```powershell
$env:REDDIT_USER_AGENT = 'script:is5126-group6:v1.0 (by /u/YOUR_REDDIT_USERNAME)'
$env:REDDIT_CLIENT_ID = 'YOUR_APPROVED_CLIENT_ID'
$env:REDDIT_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
```

Alternatively set `REDDIT_ACCESS_TOKEN` with an existing valid OAuth bearer token
and `REDDIT_USER_AGENT`; the token takes precedence over client credentials.
Expired tokens cause a clear error; supply a fresh token before retrying.
No `.env` loader or additional package is required. A `.env` file by itself is
not read by the script.

From the repository root:

```bash
python scripts/collect_reddit.py collect --limit 500
```

This writes `dataset/reddit/candidates.csv`. The collector reads the recent
r/jobs comment listing (up to 10 pages of 100 records). It may return fewer than
the requested number when the listing ends or filtering removes records; it
reports the shortfall. It cannot fetch arbitrary historical months. Recent
comments may contain very few toxic examples, so 500 candidates do not guarantee
50 toxic comments. Extend collection on another date or use a permitted broader
export if necessary; never change a label to reach the quota. Choose a new output
filename for additional batches, merge the raw batches, then run `import-csv` to
deduplicate them before annotation. Import resets all annotation fields, so do
not use it to merge already annotated work.

Without OAuth, prepare a CSV from an export that your access permits, then run:

```bash
python scripts/collect_reddit.py import-csv --input dataset/reddit/approved_export.csv
```

The export must have `comment_id,text,created_utc,subreddit`, with `subreddit=jobs`
and `created_utc` as Unix seconds (convert ISO dates or milliseconds before import).
Optional metadata columns use the names below. The importer discards other columns,
including author details, and creates a fresh blank annotation sheet. It filters
deleted/removed bodies and duplicate IDs/text. Export sources should already exclude
bot comments; the API collector also filters AutoModerator. Other bots, non-English
text and unusable context must be excluded during review.

### 2. Annotate the candidate sheet

Open the UTF-8 CSV in a CSV-aware editor. Keep IDs as text and save as UTF-8 CSV;
quotes, commas and multiline comments must be preserved. Do not edit comment text
or metadata. Follow `permalink` to inspect the parent/thread when needed. If an
export lacks links, retain source access for contextual review. Inspect raw text;
the collector deliberately does not call `cleanup_text` or lowercase the data.

| Column | Meaning / action |
|---|---|
| `comment_id` | Reddit comment ID, without `t1_`; stable join key |
| `text` | Original comment body; downstream uses the existing preprocessing |
| `created_utc` | Comment creation time, Unix seconds in UTC |
| `subreddit` | Always `jobs` |
| `link_id`, `parent_id`, `permalink` | Optional thread/parent references and source link |
| `collected_utc` | UTC acquisition timestamp; import time if the export omits it |
| `is_toxic` | Fill `0` (non-toxic), `1` (toxic), or `uncertain`; initially blank |
| `annotation_status` | `pending`, `reviewed`, or `excluded`; initially `pending` |
| `annotator` | Team reviewer code, e.g. `A1`; required for reviewed rows |
| `annotation_notes` | Brief reason, context or resolution of disagreement |

Use this working rubric for the proposal's binary toxicity labels, independently
of whether the comment sounds positive or negative:

- **0, non-toxic:** civil advice, support, factual disagreement or criticism of a
  situation without abusive attacks. Complaining about layoffs or rejection is
  not automatically toxic.
- **1, toxic:** insulting or demeaning attacks, harassment, threats, identity-based
  hate, or abusive language directed at people/groups. Judge tone and intent in
  context; profanity by itself, quotations and sarcasm need careful review.
- **uncertain:** unclear target, sarcasm or missing context. Leave `pending` for a
  second reviewer or mark `excluded`. Do not force uncertain cases into a class.

Illustrative examples below are synthetic annotation guidance, not Reddit data:
“The job search has been frustrating” is non-toxic despite negative sentiment;
“You are worthless and deserve to fail” is toxic because it attacks a person.

Prefer two independent reviewers; discuss disagreements and record the final
decision in `annotation_notes` before marking `reviewed`. The script checks that
labels and reviewer IDs are filled, but cannot verify that a human actually read
the comment. Do not use model predictions, keywords or Reddit scores as gold labels.
Exclude non-English, bot-generated, deleted or uninterpretable comments. Check
source availability during review and again before reporting results, and remove
content deleted at the source in accordance with the applicable access rules.
Keep additional raw exports, review sheets and credentials out of Git. The
explicitly approved 100-row AI sample and its manifest are the only Reddit
data-sharing exceptions in this repository.

### 3. Export the balanced test set

After at least 50 reviewed rows in each class:

```bash
python scripts/collect_reddit.py finalize
# Optional explicit paths (existing output files are never overwritten):
python scripts/collect_reddit.py finalize --input dataset/reddit/candidates.csv --output dataset/reddit/test_100_v2.csv --seed 42
```

The output has **exactly 100 distinct comments: 50 with `is_toxic=0`, 50 with
`is_toxic=1`**. It retains annotation metadata and the shared comment-table
interface `comment_id,text,created_utc`. Duplicate IDs/text, malformed rows,
invalid labels and insufficient reviewed classes cause an error before export.
Pending/excluded rows are not eligible. Sorting by ID before seeded sampling makes
the result independent of spreadsheet row order. Blank/uncertain labels never
become zero automatically.

For downstream evaluation, read `text` and `is_toxic`. Apply the model's existing
lowercase → `cleanup_text` → saved training tokenizer → same padding/truncation
pipeline; do not refit anything on Reddit. Keep the threshold chosen from training
out-of-fold predictions. Report precision, recall, macro F1 and PR-AUC. Balanced
sampling changes class prevalence, so precision/PR-AUC are conditional on this
test design, not estimates for live r/jobs traffic. With 100 rows, a few comments
can change scores materially. Do not use this curated set to estimate monthly
topic frequencies or toxicity rates; that needs a separate time-based corpus.
