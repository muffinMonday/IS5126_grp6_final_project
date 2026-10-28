"""Run from any directory: python /path/to/repo/scripts/collect_reddit.py --help."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from functions.reddit_collection import (
  RedditAPIError, RedditClient, balanced_sample, collect_comments, prepare_comments,
  ai_draft, read_browser_comments, read_comments, write_comments,
)


def main(argv=None):
  parser = argparse.ArgumentParser(
    description='Prepare 50 non-toxic + 50 toxic, manually reviewed r/jobs comments.',
  )
  commands = parser.add_subparsers(dest='command', required=True)
  collect = commands.add_parser('collect', help='Fetch recent candidates using approved OAuth access.')
  collect.add_argument('--limit', type=int, default=500, help='Candidate target, 1..1000 (default: 500).')
  collect.add_argument('--max-pages', type=int, default=10, help='At most 10 pages of 100 API records.')
  collect.add_argument('--output', type=Path, default=ROOT / 'dataset/reddit/candidates.csv')

  local = commands.add_parser('import-csv', help='Prepare a blank sheet from a permitted r/jobs export.')
  local.add_argument('--input', type=Path, required=True)
  local.add_argument('--output', type=Path, default=ROOT / 'dataset/reddit/candidates.csv')

  browser = commands.add_parser('import-browser', help='Import local rendered-comment JSON snapshots.')
  browser.add_argument('--input', type=Path, nargs='+', required=True)
  browser.add_argument('--output', type=Path, default=ROOT / 'dataset/reddit/browser_candidates.csv')

  draft = commands.add_parser('ai-draft', help='Attach AI suggestions; keep all 100 rows pending review.')
  draft.add_argument('--input', type=Path, required=True)
  draft.add_argument('--annotations', type=Path, required=True)
  draft.add_argument('--output', type=Path, default=ROOT / 'dataset/reddit/test_100_ai_draft.csv')
  draft.add_argument('--seed', type=int, default=42)

  finalize = commands.add_parser('finalize', help='Export exactly 50 reviewed comments from each class.')
  finalize.add_argument('--input', type=Path, default=ROOT / 'dataset/reddit/candidates.csv')
  finalize.add_argument('--output', type=Path, default=ROOT / 'dataset/reddit/test_100.csv')
  finalize.add_argument('--seed', type=int, default=42)
  args = parser.parse_args(argv)

  try:
    if args.output.exists():
      raise ValueError(f'Output already exists: {args.output}. Choose a new filename.')
    if args.command == 'collect':
      if not 1 <= args.limit <= 1000 or not 1 <= args.max_pages <= 10:
        raise ValueError('Use --limit 1..1000 and --max-pages 1..10.')
      rows = collect_comments(RedditClient.from_env(), args.limit, args.max_pages)
      if len(rows) < args.limit:
        print(f'Listing ended or page budget reached: {len(rows)}/{args.limit} candidates.',
              file=sys.stderr)
    elif args.command == 'import-csv':
      rows = prepare_comments(read_comments(args.input))
    elif args.command == 'import-browser':
      rows = read_browser_comments(args.input)
    elif args.command == 'ai-draft':
      annotations = json.loads(args.annotations.read_text(encoding='utf-8'))
      rows = ai_draft(read_comments(args.input), annotations, seed=args.seed)
    else:
      rows = balanced_sample(read_comments(args.input), seed=args.seed)
    if not rows:
      raise ValueError('No usable comments; no output written.')
    write_comments(args.output, rows)
  except (OSError, ValueError, RedditAPIError) as exc:
    parser.exit(2, f'Error: {exc}\n')

  print(f'Wrote {len(rows)} comments to {args.output}')
  if args.command == 'finalize':
    print('Verified: 50 non-toxic (0), 50 toxic (1), unique and human-reviewed.')
    print('Use only for final zero-shot evaluation, never training or threshold tuning.')
  elif args.command == 'ai-draft':
    print('50 non-toxic + 50 toxic AI suggestions; ALL rows remain pending human review.')
    print('These are preliminary labels, not a human-labelled gold test set.')
  else:
    print('Review raw text/context; fill is_toxic, annotator, and annotation_status=reviewed.')
    print('Leave unclear cases pending or excluded. See dataset/README.md for the rubric.')


if __name__ == '__main__':
  main()
