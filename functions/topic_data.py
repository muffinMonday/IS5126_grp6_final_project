"""Historical topic corpus preparation; does not change toxicity preprocessing."""

import hashlib
import html
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from functions.preprocessing import cleanup_text


REQUIRED = {'comment_id', 'text', 'created_utc', 'subreddit'}
RAW_FIELDS = ['comment_id', 'text', 'created_utc', 'subreddit', 'link_id',
              'parent_id', 'permalink', 'collected_utc', 'is_bot', 'language',
              'sampling_stratum']
ARCHIVE_BASE = 'https://arctic-shift.photon-reddit.com/api/comments/search'


def month_windows(year=2026, as_of=None):
  """UTC half-open windows; include the current partial month, never future months."""
  as_of = pd.Timestamp(as_of or datetime.now(timezone.utc))
  if as_of.tzinfo is None:
    raise ValueError('as_of must include a timezone.')
  as_of = as_of.tz_convert('UTC')
  start = pd.Timestamp(year=year, month=1, day=1, tz='UTC')
  windows = []
  while start < as_of and start.year == year:
    next_month = start + pd.offsets.MonthBegin(1)
    end = min(next_month, as_of)
    windows.append({'month': start.strftime('%Y-%m'), 'start': start,
                    'end': end, 'partial_month': end < next_month})
    start = next_month
  return windows


def clean_topic_comments(records, year=2026, as_of=None, min_words=3):
  """Keep raw text, deduplicate IDs only, and record one exclusion reason per row."""
  if min_words < 1:
    raise ValueError('min_words must be positive.')
  frame = pd.DataFrame(records)
  if not REQUIRED.issubset(frame.columns):
    raise ValueError(f'Missing columns: {sorted(REQUIRED - set(frame.columns))}')
  cutoff = pd.Timestamp(as_of or datetime.now(timezone.utc))
  if cutoff.tzinfo is None:
    raise ValueError('as_of must include a timezone.')
  cutoff = cutoff.tz_convert('UTC')
  start = pd.Timestamp(year=year, month=1, day=1, tz='UTC')
  end = min(pd.Timestamp(year=year + 1, month=1, day=1, tz='UTC'), cutoff)
  work = frame.reindex(columns=RAW_FIELDS).copy()
  work['source_row'] = range(len(work))
  work['comment_id'] = work['comment_id'].fillna('').astype(str).str.strip().str.lower().str.removeprefix('t1_')
  work['subreddit'] = work['subreddit'].fillna('').astype(str).str.strip().str.lower()
  seconds = pd.to_numeric(work['created_utc'], errors='coerce')
  work['created_at'] = pd.to_datetime(seconds, unit='s', utc=True, errors='coerce')
  work['month'] = work['created_at'].dt.strftime('%Y-%m')
  work['exclusion_reason'] = ''

  def exclude(mask, reason):
    work.loc[mask & work['exclusion_reason'].eq(''), 'exclusion_reason'] = reason

  exclude(~work['comment_id'].str.fullmatch('[a-z0-9]+'), 'invalid_comment_id')
  exclude(work['subreddit'].ne('jobs'), 'wrong_subreddit')
  exclude(work['created_at'].isna(), 'invalid_timestamp')
  exclude(work['created_at'].ge(cutoff), 'at_or_after_cutoff')
  exclude(work['created_at'].lt(start) | work['created_at'].ge(end), 'outside_requested_period')
  valid_text = work['text'].map(lambda value: isinstance(value, str) and bool(value.strip()))
  exclude(~valid_text, 'empty_or_nonstring_text')
  raw_keys = work['text'].map(lambda value: value.strip().casefold() if isinstance(value, str) else '')
  exclude(raw_keys.isin(['[deleted]', '[removed]']), 'deleted_or_removed')
  bot = work['is_bot'].fillna('').astype(str).str.lower().isin(['true', '1', 'yes'])
  if 'author' in frame:
    bot |= frame['author'].fillna('').astype(str).str.casefold().eq('automoderator')
  exclude(bot, 'known_bot')
  language = work['language'].fillna('').astype(str).str.strip().str.lower()
  exclude(~language.isin(['', 'en', 'eng', 'english', 'unknown']), 'known_non_english')

  # Conflicting snapshots must be resolved before choosing which version to keep.
  possible = work.loc[work['exclusion_reason'].eq('')]
  for comment_id, group in possible.groupby('comment_id'):
    if len(group[['text', 'created_at']].drop_duplicates()) > 1:
      raise ValueError(f'Conflicting snapshots for comment_id={comment_id}; resolve source records.')
  duplicates = possible['comment_id'].duplicated()
  exclude(work.index.isin(possible.index[duplicates]), 'duplicate_comment_id')

  work['text_topic'] = work['text'].map(
    lambda value: cleanup_text(html.unescape(value).lower()) if isinstance(value, str) else '')
  work['word_count'] = work['text_topic'].map(lambda value: len(re.findall(r'[a-z]+', value)))
  exclude(work['text_topic'].eq(''), 'empty_after_cleaning')
  exclude(work['word_count'].lt(min_words), 'too_short')
  work['language_status'] = language.map(lambda value: 'unverified' if value in ('', 'unknown') else 'source_english')
  work['normalized_text_key'] = work['text_topic'].str.casefold().str.split().str.join(' ')
  eligible = work.loc[work['exclusion_reason'].eq('')].copy()
  eligible['duplicate_text'] = eligible['normalized_text_key'].duplicated(keep=False)
  excluded = work.loc[work['exclusion_reason'].ne('')].copy()
  audit = (work['exclusion_reason'].replace('', 'eligible').value_counts()
           .rename_axis('reason').reset_index(name='n_rows'))
  return eligible.reset_index(drop=True), excluded.reset_index(drop=True), audit


def sample_months(eligible, year=2026, as_of=None, per_month=1000, seed=42,
                  source_complete=None, strata_per_month=None):
  """Sample without replacement from each supplied eligible monthly population."""
  if per_month < 1:
    raise ValueError('per_month must be positive.')
  source_complete = source_complete or {}
  samples, report = [], []
  for window in month_windows(year, as_of):
    month = window['month']
    pool = eligible.loc[eligible['month'].eq(month)].sort_values('comment_id')
    if strata_per_month:
      parts = []
      for stratum in range(strata_per_month):
        quota = per_month // strata_per_month + (stratum < per_month % strata_per_month)
        group = pool.loc[pd.to_numeric(pool['sampling_stratum'], errors='coerce').eq(stratum)]
        parts.append(group.sample(n=min(quota, len(group)), random_state=seed))
      selected = pd.concat(parts)
    else:
      selected = pool.sample(n=min(per_month, len(pool)), random_state=seed)
    samples.append(selected)
    complete = source_complete.get(month)
    report.append({'month': month, 'window_start_utc': window['start'].isoformat(),
                   'window_end_utc': window['end'].isoformat(),
                   'partial_month': window['partial_month'], 'target': per_month,
                   'eligible_pool': len(pool), 'selected': len(selected),
                   'shortfall': max(0, per_month - len(selected)),
                   'source_window_complete': complete,
                   'status': 'target_met' if len(selected) == per_month else 'insufficient_supplied_data'})
  sample = pd.concat(samples, ignore_index=True) if samples else eligible.iloc[:0].copy()
  return sample, pd.DataFrame(report)


class ArchiveUnavailable(RuntimeError):
  pass


class ArcticShiftClient:
  """Bounded public archive requests; errors stop collection instead of skipping time."""

  def __init__(self, max_requests=100, max_seconds=90, interval=0.5):
    self.max_requests = max_requests
    self.max_seconds = max_seconds
    self.interval = interval
    self.requests = 0
    self.started = None

  def search(self, start, end):
    if self.started is None:
      self.started = time.monotonic()
    if self.requests >= self.max_requests or time.monotonic() - self.started >= self.max_seconds:
      raise ArchiveUnavailable('Collection budget reached; this window is incomplete.')
    if self.requests:
      time.sleep(self.interval)
    params = {'subreddit': 'jobs', 'after': start - 1, 'before': end,
              'limit': 100, 'sort': 'asc',
              'fields': 'id,body,created_utc,subreddit,link_id,parent_id,author'}
    request = urllib.request.Request(ARCHIVE_BASE + '?' + urllib.parse.urlencode(params),
                                    headers={'User-Agent': 'IS5126-Group6-topics/1.0'})
    self.requests += 1
    try:
      with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    except urllib.error.HTTPError as exc:
      retry = exc.headers.get('Retry-After', '') if exc.headers else ''
      code = exc.code
      exc.close()
      raise ArchiveUnavailable(f'Archive HTTP {code}; Retry-After={retry or "unspecified"}. No fallback or automatic retry.') from None
    except (OSError, ValueError) as exc:
      raise ArchiveUnavailable(f'Archive request failed: {type(exc).__name__}.') from None
    rows = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
      raise ArchiveUnavailable('Unexpected archive response; expected data list.')
    return rows


def fetch_archive_window(client, start, end):
  """Split saturated intervals rather than silently taking the first 100 comments."""
  start, end = int(start), int(end)
  rows = client.search(start, end)
  if len(rows) >= 100:
    if end - start <= 1:
      raise ArchiveUnavailable('At least 100 comments in one second; completeness cannot be established.')
    middle = (start + end) // 2
    return (fetch_archive_window(client, start, middle)
            + fetch_archive_window(client, middle, end))
  for row in rows:
    value = row.get('created_utc')
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not start <= value < end:
      raise ArchiveUnavailable('Archive returned a record outside the requested time interval.')
    if str(row.get('subreddit', '')).lower() != 'jobs':
      raise ArchiveUnavailable('Archive returned a different subreddit.')
  return rows


def archive_records(rows, collected_utc):
  """Map source fields, retaining bot flags without storing author names."""
  return [{'comment_id': row.get('id'), 'text': row.get('body'),
           'created_utc': row.get('created_utc'), 'subreddit': row.get('subreddit'),
           'link_id': row.get('link_id', ''), 'parent_id': row.get('parent_id', ''),
           'permalink': row.get('permalink') or
             f'https://www.reddit.com/r/jobs/comments/{str(row.get("link_id", "")).removeprefix("t3_")}/_/{row.get("id", "")}/',
           'collected_utc': collected_utc,
           'is_bot': str(row.get('author', '')).casefold() == 'automoderator',
           'language': ''} for row in rows]


def export_topic_run(output_dir, records, year, as_of, source, per_month=1000,
                     seed=42, source_complete=None, min_words=3, strata_per_month=None):
  """Write a new local run with sample, exclusions, provenance and coverage."""
  output_dir = Path(output_dir)
  if output_dir.exists():
    raise FileExistsError(f'Choose a new run directory: {output_dir}')
  eligible, excluded, audit = clean_topic_comments(records, year, as_of, min_words)
  sample, coverage = sample_months(eligible, year, as_of, per_month, seed,
                                  source_complete, strata_per_month)
  output_dir.mkdir(parents=True)
  outputs = {'comments_clean.csv': sample, 'eligible_pool.csv': eligible,
             'excluded_rows.csv': excluded, 'cleaning_audit.csv': audit,
             'monthly_coverage.csv': coverage}
  # Store only the corpus schema; do not copy toxicity labels or author details.
  raw = pd.DataFrame(records).reindex(columns=RAW_FIELDS)
  if 'author' in pd.DataFrame(records):
    known_bot = pd.DataFrame(records)['author'].fillna('').astype(str).str.casefold().eq('automoderator')
    raw['is_bot'] = raw['is_bot'].fillna('').astype(str).str.lower().isin(['true', '1', 'yes']) | known_bot
  outputs['raw_candidates.csv'] = raw
  for name, frame in outputs.items():
    frame.to_csv(output_dir / name, index=False, encoding='utf-8-sig')
  manifest = {'year': year, 'as_of_utc': pd.Timestamp(as_of).isoformat(), 'seed': seed,
              'target_per_month': per_month, 'min_words': min_words, 'source': source,
              'sampling': 'Seeded sampling without replacement after cleaning; IDs sorted within month.',
              'population': 'Supplied eligible records per month; archive coverage does not imply full Reddit coverage.',
              'limitations': ['Equal monthly sample sizes do not measure absolute discussion volume.',
                             'Current month is partial; missing months are not zero activity.',
                             'Language is unverified unless supplied by source; inspect a random sample.',
                             'Known bots only are filtered; unknown bots may remain.',
                             'Different IDs with matching text are retained and flagged.'],
              'counts': {'raw': len(raw), 'eligible': len(eligible), 'excluded': len(excluded),
                         'selected': len(sample)},
              'sha256': {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                         for name in outputs}}
  if strata_per_month:
    manifest['sampling'] = ('Equal quotas across equal-duration UTC time strata; seeded random '
                            'anchors define bounded candidate batches, followed by cleaning and '
                            'sampling without replacement within each stratum.')
    manifest['strata_per_month'] = strata_per_month
    manifest['limitations'].append('Random-anchor candidate batches are not a simple random sample of all monthly comments; inclusion probabilities depend on local activity.')
  (output_dir / 'sampling_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
  return sample, coverage, manifest
