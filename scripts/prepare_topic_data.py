"""Prepare monthly Objective 1 samples from an archive or a local CSV export."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from functions.reddit_collection import read_comments
from functions.topic_data import (
  ArchiveUnavailable, ArcticShiftClient, archive_records, export_topic_run,
  fetch_archive_window, month_windows,
)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('mode', choices=['archive', 'import-csv'])
  parser.add_argument('--input', type=Path)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--year', type=int, default=2026)
  parser.add_argument('--per-month', type=int, default=1000)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--min-words', type=int, default=3)
  parser.add_argument('--as-of', default=datetime.now(timezone.utc).isoformat())
  parser.add_argument('--max-requests', type=int, default=100)
  parser.add_argument('--max-seconds', type=int, default=90)
  args = parser.parse_args(argv)
  if args.output_dir.exists():
    parser.error('Output directory exists; choose a new run name.')
  if args.per_month < 1 or args.max_requests < 1 or not 0 < args.max_seconds <= 120:
    parser.error('Use positive limits and a time budget no greater than 120 seconds.')
  if args.mode == 'import-csv':
    if args.input is None:
      parser.error('import-csv requires --input.')
    rows = read_comments(args.input)
    sample, coverage, _ = export_topic_run(
      args.output_dir, rows, args.year, args.as_of,
      {'method': 'local_csv', 'file': args.input.name, 'coverage': 'not independently verified'},
      args.per_month, args.seed, min_words=args.min_words)
    print(coverage.to_string(index=False))
    print(f'Prepared {len(sample)} comments in {args.output_dir}')
    return 0

  windows = month_windows(args.year, args.as_of)
  client = ArcticShiftClient(args.max_requests, args.max_seconds)
  records, complete, status = [], {}, []
  error = None
  for window in windows:
    entry = {'month': window['month'], 'partial_month': window['partial_month'],
             'window_start_utc': window['start'].isoformat(),
             'window_end_utc': window['end'].isoformat(), 'status': 'not_attempted',
             'records_retrieved': None}
    if error is None:
      try:
        rows = fetch_archive_window(client, window['start'].timestamp(), window['end'].timestamp())
        records.extend(archive_records(rows, datetime.now(timezone.utc).isoformat()))
        complete[window['month']] = True
        entry.update(status='archive_window_exhausted', records_retrieved=len(rows))
      except ArchiveUnavailable as exc:
        error = str(exc)
        entry.update(status='source_error', error=error)
    status.append(entry)
  if records:
    export_topic_run(args.output_dir, records, args.year, args.as_of,
                     {'method': 'arctic_shift', 'requests': client.requests},
                     args.per_month, args.seed, complete, args.min_words)
  else:
    args.output_dir.mkdir(parents=True)
  report = {'year': args.year, 'as_of_utc': args.as_of, 'target_per_month': args.per_month,
            'source': 'Arctic Shift public archive', 'requests': client.requests,
            'status': 'source_error' if error else 'collection_finished',
            'months': status, 'total_retrieved': len(records),
            'note': 'Request failures and unattempted months are not zero-comment months.'}
  (args.output_dir / 'acquisition_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
  print(json.dumps(report, indent=2))
  return 2 if error else 0


if __name__ == '__main__':
  sys.exit(main())
