"""Collect time-stratified monthly topic candidates, checkpoint, clean and sample."""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from functions.topic_data import (
  ARCHIVE_BASE, ArchiveUnavailable, ArcticShiftClient, archive_records,
  clean_topic_comments, export_topic_run, month_windows,
)


def collect(work_dir, year=2026, per_month=1000, strata=20, seed=42,
            min_words=3, as_of=None):
  work_dir = Path(work_dir)
  work_dir.mkdir(parents=True, exist_ok=True)
  plan_file = work_dir / 'collection_plan.json'
  if plan_file.exists():
    plan = json.loads(plan_file.read_text(encoding='utf-8'))
    expected = (year, per_month, strata, seed, min_words)
    actual = tuple(plan[key] for key in ['year', 'per_month', 'strata', 'seed', 'min_words'])
    if expected != actual or (as_of and as_of != plan['as_of']):
      raise ValueError('Resume parameters differ from the saved collection plan.')
    as_of = plan['as_of']
  else:
    as_of = as_of or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    plan = dict(year=year, per_month=per_month, strata=strata, seed=seed,
                min_words=min_words, as_of=as_of, source=ARCHIVE_BASE)
    plan_file.write_text(json.dumps(plan, indent=2), encoding='utf-8')
  if (work_dir / 'prepared').exists():
    raise FileExistsError('Prepared output already exists; use a new work directory.')
  cache = work_dir / 'source_cache'
  cache.mkdir(exist_ok=True)
  client = ArcticShiftClient(max_requests=10000, max_seconds=7200, interval=1.0)
  records, acquisition, errors = [], [], []
  for window in month_windows(year, as_of):
    month = window['month']
    start, end = int(window['start'].timestamp()), int(window['end'].timestamp())
    for stratum in range(strata):
      left = start + (end - start) * stratum // strata
      right = start + (end - start) * (stratum + 1) // strata
      quota = per_month // strata + (stratum < per_month % strata)
      state_path = cache / f'{month}_{stratum:02d}.json'
      if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
      else:
        anchor = random.Random(f'{seed}:{month}:{stratum}').randrange(left, right)
        state = dict(month=month, stratum=stratum, start=left, end=right,
                     anchor=anchor, quota=quota, rows=[], requests=[], complete=False)
      if not state['complete']:
        try:
          eligible_count = 0
          for segment_start, segment_end in [(state['anchor'], right), (left, state['anchor'])]:
            cursor = segment_start
            for _ in range(5):
              if cursor >= segment_end or eligible_count >= quota + 10:
                break
              batch = client.search(cursor, segment_end)
              if any(r.get('subreddit', '').lower() != 'jobs'
                     or not cursor <= float(r['created_utc']) < segment_end for r in batch):
                raise ArchiveUnavailable('Source returned a record outside this stratum.')
              mapped = archive_records(batch, datetime.now(timezone.utc).isoformat())
              for row in mapped:
                row['sampling_stratum'] = stratum
              state['rows'].extend(mapped)
              state['requests'].append(dict(start=cursor, end=segment_end, returned=len(batch)))
              eligible, _, _ = clean_topic_comments(state['rows'], year, as_of, min_words)
              eligible_count = len(eligible)
              state['eligible_candidates'] = eligible_count
              state_path.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
              if len(batch) < 100:
                break
              next_cursor = max(int(r['created_utc']) for r in batch)
              if next_cursor <= cursor:
                raise ArchiveUnavailable('Pagination made no timestamp progress.')
              cursor = next_cursor
            if eligible_count >= quota + 10:
              break
          state['complete'] = eligible_count >= quota
          state_path.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
          if not state['complete']:
            raise ArchiveUnavailable(f'Only {eligible_count}/{quota} eligible candidates in {month}, stratum {stratum}.')
        except (ArchiveUnavailable, KeyError, ValueError) as exc:
          errors.append(dict(month=month, stratum=stratum, error=str(exc)))
          print(f'STOP {month} stratum={stratum}: {exc}', flush=True)
          break
      records.extend(state['rows'])
      acquisition.append({key: state[key] for key in ['month', 'stratum', 'start', 'end', 'anchor',
                                                     'quota', 'eligible_candidates', 'complete', 'requests']})
      print(f'{month} stratum {stratum + 1:02d}/{strata}: '
            f'{state["eligible_candidates"]} eligible, quota={quota}', flush=True)
    if errors:
      break
    print(f'MONTH READY {month}: {per_month} eligible samples available', flush=True)
  status = dict(status='source_error' if errors else 'candidates_ready',
                plan=plan, requests_this_session=client.requests,
                completed_strata=len(acquisition), errors=errors, strata=acquisition)
  (work_dir / 'acquisition_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
  if not errors:
    sample, coverage, manifest = export_topic_run(
      work_dir / 'prepared', records, year, as_of,
      {'method': 'arctic_shift_time_stratified', 'endpoint': ARCHIVE_BASE,
       'requests': sum(len(s['requests']) for s in acquisition),
       'provenance_file': '../acquisition_status.json', 'keywords': None,
       'permalinks': 'Constructed from source post and comment IDs, not individually fetched.'},
      per_month, seed, {w['month']: False for w in month_windows(year, as_of)},
      min_words, strata)
    if not coverage['selected'].eq(per_month).all() or not sample['comment_id'].is_unique:
      raise ValueError('Final quota or uniqueness checks failed; inspect prepared files.')
    print(coverage[['month', 'selected', 'shortfall', 'partial_month']].to_string(index=False), flush=True)
    print(f'SUCCESS: {len(sample)} clean comments; output={work_dir / "prepared"}', flush=True)
  return 2 if errors else 0


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--work-dir', type=Path, required=True)
  parser.add_argument('--year', type=int, default=2026)
  parser.add_argument('--per-month', type=int, default=1000)
  parser.add_argument('--strata', type=int, default=20)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--min-words', type=int, default=3)
  parser.add_argument('--as-of')
  args = parser.parse_args()
  if not 1 <= args.strata <= args.per_month or args.min_words < 1:
    parser.error('Use 1 <= strata <= per-month and min-words >= 1.')
  return collect(args.work_dir, args.year, args.per_month, args.strata,
                 args.seed, args.min_words, args.as_of)


if __name__ == '__main__':
  sys.exit(main())
