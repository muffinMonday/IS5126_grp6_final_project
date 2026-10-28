"""Collect r/jobs candidates and sample a manually labelled toxicity test set.

Uses only the standard library; importing this module does not load TensorFlow.
Raw text is preserved for the existing training/inference preprocessing pipeline.
"""

import base64
import csv
import json
import math
import os
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


COMMENT_FIELDS = (
  'comment_id', 'text', 'created_utc', 'subreddit', 'link_id', 'parent_id',
  'permalink', 'collected_utc', 'is_toxic', 'annotation_status', 'annotator',
  'annotation_notes',
)
REQUIRED_FIELDS = ('comment_id', 'text', 'created_utc', 'subreddit')
ANNOTATION_FIELDS = ('is_toxic', 'annotation_status', 'annotator', 'annotation_notes')


class RedditAPIError(RuntimeError):
  """An API failure that can be displayed without exposing credentials."""


class RedditClient:
  """Small read-only OAuth client for an already approved Reddit Data API app."""

  def __init__(self, user_agent, token=None):
    if not user_agent.strip():
      raise ValueError('Set REDDIT_USER_AGENT to a descriptive app/contact string.')
    self.user_agent = user_agent
    self.token = token
    self.next_request_at = 0

  @classmethod
  def from_env(cls):
    client = cls(os.environ.get('REDDIT_USER_AGENT', ''))
    client.token = os.environ.get('REDDIT_ACCESS_TOKEN', '').strip()
    if not client.token:
      client_id = os.environ.get('REDDIT_CLIENT_ID', '').strip()
      secret = os.environ.get('REDDIT_CLIENT_SECRET', '').strip()
      if not client_id or not secret:
        raise ValueError('Set REDDIT_ACCESS_TOKEN or both REDDIT_CLIENT_ID and '
                         'REDDIT_CLIENT_SECRET for approved API access; '
                         'otherwise use the import-csv command.')
      credentials = base64.b64encode(f'{client_id}:{secret}'.encode()).decode()
      request = urllib.request.Request(
        'https://www.reddit.com/api/v1/access_token',
        data=urllib.parse.urlencode({'grant_type': 'client_credentials'}).encode(),
        headers={'Authorization': f'Basic {credentials}'},
      )
      result = client._request(request)
      client.token = result.get('access_token')
      if not client.token:
        raise RedditAPIError('OAuth did not return an access token. Check app approval.')
    return client

  @staticmethod
  def _retry_delay(headers, fallback):
    value = headers.get('Retry-After')
    if value is None:
      return fallback
    try:
      return max(0, float(value))
    except ValueError:
      try:
        return max(0, parsedate_to_datetime(value).timestamp() - time.time())
      except (ValueError, TypeError, OverflowError):
        return fallback

  def _request(self, request):
    request.add_header('User-Agent', self.user_agent)
    request.add_header('Accept', 'application/json')
    for attempt in range(3):
      delay = max(0, self.next_request_at - time.monotonic())
      if delay > 30:
        raise RedditAPIError(f'Reddit requests a {math.ceil(delay)}s wait. Retry later.')
      if delay:
        time.sleep(delay)
      try:
        with urllib.request.urlopen(request, timeout=20) as response:
          result = json.load(response)
          headers = response.headers
        delay = 1.0
        try:
          if float(headers.get('X-Ratelimit-Remaining', '100')) < 1:
            delay = max(delay, float(headers.get('X-Ratelimit-Reset', '60')) + 1)
        except ValueError:
          pass
        self.next_request_at = time.monotonic() + delay
        if not isinstance(result, dict):
          raise RedditAPIError('Unexpected Reddit response; expected a JSON object.')
        return result
      except urllib.error.HTTPError as exc:
        code = exc.code
        delay = self._retry_delay(exc.headers, 2 ** (attempt + 1))
        exc.close()
        if code in (401, 403):
          raise RedditAPIError(f'Reddit HTTP {code}: check approval, token expiry, '
                               'read access and User-Agent. No anonymous fallback.') from None
        if code not in (429, 500, 502, 503, 504) or attempt == 2:
          raise RedditAPIError(f'Reddit HTTP {code}; request failed.') from None
        if not math.isfinite(delay) or delay > 30:
          raise RedditAPIError('Reddit requested a long retry delay; retry later.') from None
        self.next_request_at = time.monotonic() + max(1, delay)
      except (urllib.error.URLError, TimeoutError):
        if attempt == 2:
          raise RedditAPIError('Reddit connection failed after three attempts.') from None
        self.next_request_at = time.monotonic() + 2 ** (attempt + 1)
      except (ValueError, UnicodeError):
        raise RedditAPIError('Reddit returned invalid JSON.') from None

  def comments(self, after=None):
    params = {'limit': 100, 'raw_json': 1}
    if after:
      params['after'] = after
    request = urllib.request.Request(
      'https://oauth.reddit.com/r/jobs/comments?' + urllib.parse.urlencode(params),
      headers={'Authorization': f'Bearer {self.token}'},
    )
    return self._request(request)


def _text_key(text):
  return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _normalise_comment(record):
  missing = [key for key in REQUIRED_FIELDS if record.get(key) in (None, '')]
  if missing:
    raise ValueError(f'Comment is missing required values: {", ".join(missing)}')
  row = {key: record.get(key, '') for key in COMMENT_FIELDS}
  comment_id = str(row['comment_id']).strip().lower()
  row['comment_id'] = comment_id.removeprefix('t1_')
  if not re.fullmatch(r'[a-z0-9]+', row['comment_id']):
    raise ValueError('comment_id must be a Reddit comment ID (optionally t1_ prefixed).')
  if str(row['subreddit']).strip().casefold() != 'jobs':
    raise ValueError('Only r/jobs comments belong in this test set.')
  row['subreddit'] = 'jobs'
  if not isinstance(row['text'], str) or not row['text'].strip():
    raise ValueError(f'Comment {row["comment_id"]} has empty text.')
  try:
    timestamp = float(row['created_utc'])
    if not math.isfinite(timestamp) or not 0 < timestamp <= time.time() + 86400:
      raise ValueError
    row['created_utc'] = int(timestamp)
  except (TypeError, ValueError, OverflowError):
    raise ValueError('created_utc must be Unix seconds, positive and not in the future.') from None
  return row


def prepare_comments(records):
  """Create a blank annotation sheet from permitted exports; deduplicate ID/text."""
  rows, seen_ids, seen_texts = [], set(), set()
  collected = datetime.now(timezone.utc).isoformat()
  for record in records:
    if str(record.get('text', '')).strip().casefold() in ('[deleted]', '[removed]'):
      continue
    row = _normalise_comment(record)
    text_key = _text_key(row['text'])
    if row['comment_id'] in seen_ids or text_key in seen_texts:
      continue
    seen_ids.add(row['comment_id'])
    seen_texts.add(text_key)
    for field in ANNOTATION_FIELDS:
      row[field] = ''
    row['annotation_status'] = 'pending'
    row['collected_utc'] = row['collected_utc'] or collected
    rows.append(row)
  return rows


def collect_comments(client, limit=500, max_pages=10):
  """Collect a bounded recent candidate pool, not a historical or balanced sample."""
  if not 1 <= limit <= 1000 or not 1 <= max_pages <= 10:
    raise ValueError('Use limit 1..1000 and max_pages 1..10 for this small collection.')
  records, cursors = [], set()
  after = None
  for _ in range(max_pages):
    result = client.comments(after)
    data = result.get('data')
    if not isinstance(data, dict) or not isinstance(data.get('children'), list):
      raise RedditAPIError('Unexpected Reddit listing; expected data.children.')
    for child in data['children']:
      if child.get('kind') != 't1':
        continue
      comment = child.get('data', {})
      if (not isinstance(comment.get('body'), str) or not comment['body'].strip()
          or str(comment.get('author', '')).casefold() == 'automoderator'
          or comment.get('removed_by_category')):
        continue
      permalink = comment.get('permalink', '')
      if permalink.startswith('/r/'):
        permalink = 'https://www.reddit.com' + permalink
      records.append({
        'comment_id': comment.get('id'), 'text': comment.get('body'),
        'created_utc': comment.get('created_utc'), 'subreddit': comment.get('subreddit'),
        'link_id': comment.get('link_id', ''), 'parent_id': comment.get('parent_id', ''),
        'permalink': permalink,
      })
    rows = prepare_comments(records)
    if len(rows) >= limit:
      return rows[:limit]
    after = data.get('after')
    if not after or after in cursors or not data['children']:
      break
    cursors.add(after)
  return prepare_comments(records)


def read_comments(path):
  with Path(path).open(encoding='utf-8-sig', newline='') as stream:
    reader = csv.DictReader(stream, strict=True)
    if len(reader.fieldnames or ()) != len(set(reader.fieldnames or ())):
      raise ValueError('CSV has duplicate column names.')
    missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or ())
    if missing:
      raise ValueError(f'CSV is missing columns: {", ".join(sorted(missing))}')
    try:
      rows = list(reader)
    except csv.Error as exc:
      raise ValueError(f'Invalid CSV: {exc}') from None
    if any(None in row or any(value is None for value in row.values()) for row in rows):
      raise ValueError('CSV row widths do not match the header; check quoting and commas.')
  if not rows:
    raise ValueError('CSV contains no comments.')
  return rows


def read_browser_comments(paths):
  """Import local snapshots of rendered comments; never drive a browser or log in."""
  records = []
  for path in paths:
    path = Path(path)
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(snapshot, list) or any(not isinstance(r, dict) for r in snapshot):
      raise ValueError('Browser snapshot must be a JSON array of comment objects.')
    captured = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    for record in snapshot:
      try:
        created = datetime.fromisoformat(record['created_iso'])
        if created.tzinfo is None:
          raise ValueError
      except (KeyError, TypeError, ValueError):
        raise ValueError('Browser comments require created_iso with an explicit timezone.') from None
      row = {**record, 'created_utc': created.timestamp(), 'collected_utc': captured}
      row = _normalise_comment(row)
      if not isinstance(row['permalink'], str) or not row['permalink']:
        raise ValueError('Browser comments require a source permalink.')
      url = urllib.parse.urlparse(urllib.parse.urljoin('https://www.reddit.com', row['permalink']))
      parts = url.path.strip('/').split('/')
      if (url.scheme != 'https' or url.netloc != 'www.reddit.com'
          or len(parts) != 6 or parts[:3] != ['r', 'jobs', 'comments']
          or parts[-1] != row['comment_id'] or row['link_id'] != 't3_' + parts[3]
          or url.query or url.fragment):
        raise ValueError('Browser permalink must match the r/jobs post and comment IDs.')
      row['permalink'] = url.geturl()
      records.append(row)
  return prepare_comments(records)


def ai_draft(records, annotations, seed=42):
  """Attach explicit AI suggestions to 100 source comments; keep every row pending."""
  if not isinstance(annotations, dict) or len(annotations) != 100:
    raise ValueError('AI annotations must map exactly 100 comment IDs to label/reason objects.')
  sources = {row['comment_id']: row for row in prepare_comments(records)}
  rows = []
  for comment_id, annotation in sorted(annotations.items()):
    if comment_id not in sources:
      raise ValueError(f'AI annotation has no matching source comment: {comment_id}')
    if (not isinstance(annotation, dict) or type(annotation.get('label')) is not int
        or annotation['label'] not in (0, 1)
        or not isinstance(annotation.get('reason'), str) or not annotation['reason'].strip()):
      raise ValueError('Every AI suggestion needs integer label 0/1 and a nonempty reason.')
    row = sources[comment_id].copy()
    row.update(is_toxic=str(annotation['label']), annotation_status='pending',
               annotator='AI:Codex',
               annotation_notes='AI preliminary; human review required. ' + annotation['reason'])
    rows.append(row)
  if sum(row['is_toxic'] == '1' for row in rows) != 50:
    raise ValueError('AI draft requires exactly 50 suggested labels in each class.')
  random.Random(seed).shuffle(rows)
  return rows


def write_comments(path, rows):
  """Never overwrite raw data or an existing annotation sheet."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('x', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=COMMENT_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def balanced_sample(records, per_class=50, seed=42):
  """Select human-reviewed 0/1 labels only; fail rather than pad a missing class."""
  if per_class < 1:
    raise ValueError('per_class must be positive.')
  groups = {'0': [], '1': []}
  seen_ids, seen_texts = set(), set()
  for record in records:
    row = _normalise_comment(record)
    text_key = _text_key(row['text'])
    if row['comment_id'] in seen_ids or text_key in seen_texts:
      raise ValueError('Duplicate comment ID or text in annotation file; resolve before export.')
    seen_ids.add(row['comment_id'])
    seen_texts.add(text_key)
    status = str(row['annotation_status']).strip().lower()
    label = str(row['is_toxic']).strip()
    if status not in ('pending', 'reviewed', 'excluded'):
      raise ValueError('annotation_status must be pending, reviewed or excluded.')
    if label not in ('', '0', '1', 'uncertain'):
      raise ValueError('is_toxic must be blank, 0, 1 or uncertain; never use sentiment labels.')
    if status != 'reviewed':
      continue
    if label not in groups or not str(row['annotator']).strip():
      raise ValueError('Every reviewed comment needs is_toxic 0/1 and an annotator ID.')
    if str(row['annotator']).strip().casefold().startswith('ai:'):
      raise ValueError('AI suggestions need human review and a human annotator ID before finalize.')
    if text_key in ('[deleted]', '[removed]'):
      raise ValueError('Deleted/removed comments cannot be included in the test set.')
    row.update(is_toxic=label, annotation_status=status)
    groups[label].append(row)
  if any(len(group) < per_class for group in groups.values()):
    raise ValueError(f'Need {per_class} reviewed comments per class; found '
                     f'{len(groups["0"])} non-toxic and {len(groups["1"])} toxic. '
                     'Collect/annotate more; do not relabel comments to fill a quota.')
  rng = random.Random(seed)
  sample = []
  for label in ('0', '1'):
    sample.extend(rng.sample(sorted(groups[label], key=lambda row: row['comment_id']), per_class))
  rng.shuffle(sample)
  return sample
