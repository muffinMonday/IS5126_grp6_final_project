"""Rebuild and verify frozen layoff counts from the shared archive matches.

This replacement does not re-query the live archive: the original downloader
source was lost. The original monthly population counts and provenance are
retained, so the published notebook can be reproduced without network access.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from functions.layoff_mentions import LAYOFF_TERMS, layoff_terms


def rebuild(work_dir):
  """Recount unique non-bot matches, keeping the archived denominators."""
  work_dir = Path(work_dir)
  run = json.loads((work_dir / 'run.json').read_text(encoding='utf-8'))
  original = pd.read_csv(work_dir / 'monthly_counts.csv', dtype={'month': str})
  if not original['month'].is_unique or set(original['month']) != set(run['months']):
    raise ValueError('Monthly counts and run manifest disagree.')
  terms = run['terms']
  for term in terms:
    if run['patterns'][term] != LAYOFF_TERMS[term][1]:
      raise ValueError(f'The matching rule changed for {term}.')
  rows = []
  for _, saved in original.iterrows():
    row = saved.to_dict()
    exact_ids = set()
    for term in terms:
      path = work_dir / 'matches' / f'{term}_{saved["month"]}.json'
      hits = json.loads(path.read_text(encoding='utf-8'))
      if len({hit['id'] for hit in hits}) != len(hits):
        raise ValueError(f'Duplicate archive IDs in {path}.')
      matched = []
      for hit in hits:
        exact = bool(layoff_terms(hit['body'], [term]))
        if exact != hit['exact']:
          raise ValueError(f'Saved exact flag differs for {hit["id"]}.')
        if exact and not hit.get('is_bot', False):
          matched.append(hit['id'])
      row[f'{term}_hits'] = len(hits)
      row[f'{term}_exact'] = len(matched)
      exact_ids.update(matched)
    row['any_term_exact'] = len(exact_ids)
    rows.append(row)
  return original, pd.DataFrame(rows, columns=original.columns)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--work-dir', type=Path,
                      default=ROOT / 'dataset/reddit/layoff_mentions')
  parser.add_argument('--output', type=Path,
                      help='Optional new CSV path; existing files are never overwritten.')
  args = parser.parse_args()
  original, rebuilt = rebuild(args.work_dir)
  pd.testing.assert_frame_equal(original, rebuilt, check_dtype=False)
  if args.output:
    with args.output.open('x', encoding='utf-8', newline='') as stream:
      rebuilt.to_csv(stream, index=False)
  print(f'Verified {len(rebuilt)} months against frozen archive matches; no network requests.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
