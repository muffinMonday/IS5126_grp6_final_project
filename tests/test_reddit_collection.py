"""Offline contract checks; all comments here are synthetic, never Reddit data."""

import io
import json
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

from functions.reddit_collection import (
  COMMENT_FIELDS, RedditAPIError, RedditClient, balanced_sample, collect_comments,
  ai_draft, prepare_comments, read_browser_comments, read_comments, write_comments,
)
from scripts.collect_reddit import main


def comment(index, **overrides):
  row = dict.fromkeys(COMMENT_FIELDS, '')
  row.update(comment_id=f'c{index}', text=f'Synthetic annotation fixture number {index}.',
             created_utc=1700000000 + index, subreddit='jobs',
             is_toxic=str(index % 2), annotation_status='reviewed', annotator='tester')
  row.update(overrides)
  return row


def listing(rows, after=None):
  children = []
  for row in rows:
    data = {**row, 'id': row['comment_id'], 'body': row['text']}
    children.append({'kind': 't1', 'data': data})
  return {'data': {'children': children, 'after': after}}


class CollectionTests(unittest.TestCase):
  def test_pagination_filtering_raw_text_and_no_author_storage(self):
    raw = '  Mixed CASE, "quotes"\nsecond line!  '
    client = Mock()
    client.comments.side_effect = [
      listing([comment(0, text=raw, author='private-name'),
               comment(1, text='[removed]'), comment(2, author='AutoModerator')], 't1_c2'),
      listing([comment(0, text=raw), comment(3), comment(4, text='[deleted]')]),
    ]
    rows = collect_comments(client, limit=10)
    self.assertEqual([r['comment_id'] for r in rows], ['c0', 'c3'])
    self.assertEqual(rows[0]['text'], raw)
    self.assertNotIn('author', rows[0])
    self.assertEqual(rows[0]['annotation_status'], 'pending')
    self.assertEqual(rows[0]['is_toxic'], '')
    self.assertEqual(client.comments.call_args_list[1].args, ('t1_c2',))

  def test_stops_on_repeated_cursor_and_page_budget(self):
    client = Mock()
    client.comments.return_value = listing([comment(0)], 't1_c0')
    self.assertEqual(len(collect_comments(client)), 1)
    self.assertEqual(client.comments.call_count, 2)
    client.reset_mock()
    collect_comments(client, max_pages=1)
    self.assertEqual(client.comments.call_count, 1)

  def test_limit_and_bad_listing(self):
    client = Mock()
    client.comments.return_value = listing([comment(i) for i in range(20)])
    self.assertEqual(len(collect_comments(client, limit=5)), 5)
    with self.assertRaises(ValueError):
      collect_comments(client, limit=1001)
    client.comments.return_value = {'error': 403}
    with self.assertRaises(RedditAPIError):
      collect_comments(client)

  def test_import_deduplicates_and_resets_labels(self):
    rows = prepare_comments([comment(1, comment_id='t1_abc', text='Hello WORLD'),
                             comment(2, text=' hello   world '),
                             comment(3, comment_id='abc')])
    self.assertEqual(len(rows), 1)
    self.assertEqual(rows[0]['comment_id'], 'abc')
    self.assertEqual(rows[0]['is_toxic'], '')
    self.assertEqual(rows[0]['annotator'], '')

  def test_invalid_source_and_timestamp_rejected(self):
    for overrides in ({'subreddit': 'other'}, {'created_utc': 'nan'},
                      {'created_utc': -1}, {'created_utc': 1700000000000},
                      {'created_utc': ''}, {'text': '  '}):
      with self.subTest(overrides=overrides), self.assertRaises(ValueError):
        prepare_comments([comment(0, **overrides)])


class SamplingTests(unittest.TestCase):
  def test_browser_import_preserves_rendered_text_and_checks_provenance(self):
    source = dict(comment_id='abc', text='Original, "case"\nTwo lines.',
                  created_iso='2024-01-01T00:00:00.123000+0000', subreddit='jobs',
                  link_id='t3_post', permalink='/r/jobs/comments/post/title/abc/')
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / 'browser.json'
      path.write_text(json.dumps([source, source]), encoding='utf-8')
      rows = read_browser_comments([path])
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]['text'], source['text'])
      self.assertEqual(rows[0]['created_utc'], 1704067200)
      self.assertEqual(rows[0]['permalink'], 'https://www.reddit.com' + source['permalink'])
      self.assertEqual(rows[0]['annotation_status'], 'pending')
      for change in ({'permalink': '/r/jobs/comments/post/title/other/'},
                     {'permalink': 'https://example.com/r/jobs/comments/post/title/abc/'},
                     {'created_iso': '2024-01-01T00:00:00'}, {'link_id': 't3_wrong'}):
        path.write_text(json.dumps([{**source, **change}]), encoding='utf-8')
        with self.subTest(change=change), self.assertRaises(ValueError):
          read_browser_comments([path])

  def test_ai_draft_is_balanced_but_cannot_be_finalized_as_human_reviewed(self):
    rows = [comment(i, collected_utc='2026-10-27T12:00:00+00:00') for i in range(100)]
    annotations = {r['comment_id']: {'label': int(r['is_toxic']), 'reason': 'Synthetic reason.'}
                   for r in rows}
    draft = ai_draft(rows, annotations)
    self.assertEqual(Counter(r['is_toxic'] for r in draft), {'0': 50, '1': 50})
    self.assertTrue(all(r['annotation_status'] == 'pending' for r in draft))
    self.assertEqual(draft, ai_draft(list(reversed(rows)), annotations))
    with self.assertRaisesRegex(ValueError, 'reviewed comments'):
      balanced_sample(draft)
    for row in draft:
      row['annotation_status'] = 'reviewed'
    with self.assertRaisesRegex(ValueError, 'human annotator'):
      balanced_sample(draft)

  def test_ai_draft_rejects_missing_sources_bad_labels_and_wrong_quotas(self):
    rows = [comment(i) for i in range(100)]
    annotations = {r['comment_id']: {'label': int(r['is_toxic']), 'reason': 'Synthetic reason.'}
                   for r in rows}
    for change in ({'label': 1, 'reason': 'wrong quota'}, {'label': True, 'reason': 'bool'},
                   {'label': 0, 'reason': ''}, {'label': '0', 'reason': 'string'}):
      with self.subTest(change=change), self.assertRaises(ValueError):
        ai_draft(rows, {**annotations, 'c0': change})
    with self.assertRaises(ValueError):
      ai_draft(rows[1:], annotations)
    with self.assertRaises(ValueError):
      ai_draft(rows, {})

  def test_malformed_csv_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / 'malformed.csv'
      for content in ('comment_id,text,created_utc,subreddit\nc1,unquoted,comma,1700000000,jobs\n',
                      'comment_id,text,created_utc,subreddit\nc1,missing fields\n',
                      'comment_id,text,created_utc,subreddit,text\nc1,a,1700000000,jobs,b\n'):
        path.write_text(content, encoding='utf-8')
        with self.subTest(content=content), self.assertRaises(ValueError):
          read_comments(path)

  def test_exact_quotas_reproducibility_and_no_input_mutation(self):
    rows = [comment(i) for i in range(140)]
    rows.append(comment(200, is_toxic='uncertain', annotation_status='pending'))
    rows.append(comment(201, annotation_status='excluded'))
    result = balanced_sample(rows)
    self.assertEqual(len(result), 100)
    self.assertEqual(Counter(row['is_toxic'] for row in result), {'0': 50, '1': 50})
    self.assertEqual(len({row['comment_id'] for row in result}), 100)
    self.assertEqual(result, balanced_sample(list(reversed(rows))))
    self.assertNotEqual(result, balanced_sample(rows, seed=7))
    self.assertEqual(rows[0], comment(0))

  def test_shortfall_pending_and_invalid_review_do_not_become_gold_labels(self):
    for changes in ({'annotation_status': 'pending'}, {'annotation_status': 'excluded'},
                    {'is_toxic': ''}, {'is_toxic': 'uncertain'}, {'annotator': ''},
                    {'is_toxic': 'positive'}, {'annotation_status': 'yes'}):
      rows = [comment(i) for i in range(100)]
      rows[0].update(changes)
      with self.subTest(changes=changes), self.assertRaises(ValueError):
        balanced_sample(rows)

  def test_duplicate_id_text_and_deleted_content_rejected(self):
    for changes in ({'comment_id': 'c1'}, {'text': comment(1)['text'].upper()},
                    {'text': '[deleted]'}, {'text': '[removed]'}):
      rows = [comment(i) for i in range(100)]
      rows[0].update(changes)
      with self.subTest(changes=changes), self.assertRaises(ValueError):
        balanced_sample(rows)

  def test_csv_roundtrip_cli_and_no_overwrite(self):
    with tempfile.TemporaryDirectory() as tmp:
      source, output = Path(tmp) / 'reviewed.csv', Path(tmp) / 'test.csv'
      rows = [comment(i) for i in range(100)]
      rows[0]['text'] = 'Mixed CASE, "quotes"\nUnicode: café 中文'
      write_comments(source, rows)
      self.assertEqual(read_comments(source)[0]['text'], rows[0]['text'])
      with patch('sys.stdout', new_callable=io.StringIO):
        main(['finalize', '--input', str(source), '--output', str(output)])
      exported = read_comments(output)
      self.assertEqual(Counter(row['is_toxic'] for row in exported), {'0': 50, '1': 50})
      before = output.read_bytes()
      with self.assertRaises(FileExistsError):
        write_comments(output, rows)
      self.assertEqual(before, output.read_bytes())

  def test_failed_cli_export_leaves_no_file(self):
    with tempfile.TemporaryDirectory() as tmp:
      source, output = Path(tmp) / 'reviewed.csv', Path(tmp) / 'test.csv'
      write_comments(source, [comment(i) for i in range(99)])
      with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as exc:
        main(['finalize', '--input', str(source), '--output', str(output)])
      self.assertEqual(exc.exception.code, 2)
      self.assertFalse(output.exists())


class APITests(unittest.TestCase):
  def response(self, payload, headers=None):
    response = io.BytesIO(json.dumps(payload).encode())
    response.headers = headers or {}
    return response

  @patch('functions.reddit_collection.time.sleep')
  @patch('functions.reddit_collection.urllib.request.urlopen')
  def test_oauth_headers_and_pagination_parameter(self, urlopen, sleep):
    urlopen.side_effect = [self.response({'access_token': 'test-token'}),
                          self.response(listing([]))]
    with patch.dict('os.environ', {'REDDIT_USER_AGENT': 'test-agent',
                                  'REDDIT_CLIENT_ID': 'test-id',
                                  'REDDIT_CLIENT_SECRET': 'test-secret'}, clear=True):
      client = RedditClient.from_env()
      client.comments('t1_abc')
    token_request = urlopen.call_args_list[0].args[0]
    self.assertEqual(token_request.data, b'grant_type=client_credentials')
    request = urlopen.call_args_list[1].args[0]
    self.assertIn('after=t1_abc', request.full_url)
    self.assertTrue(request.full_url.startswith('https://oauth.reddit.com/r/jobs/comments?'))
    self.assertEqual(request.get_header('Authorization'), 'Bearer test-token')
    self.assertEqual(request.get_header('User-agent'), 'test-agent')

  @patch('functions.reddit_collection.time.sleep')
  @patch('functions.reddit_collection.urllib.request.urlopen')
  def test_retry_429_and_honour_rate_limit(self, urlopen, sleep):
    urlopen.side_effect = [urllib.error.HTTPError('https://oauth.reddit.com', 429, 'busy',
                                                {'Retry-After': '2'}, None),
                          self.response(listing([]), {'X-Ratelimit-Remaining': '0',
                                                       'X-Ratelimit-Reset': '600'})]
    client = RedditClient('test-agent', 'test-token')
    client.comments()
    self.assertGreaterEqual(sleep.call_args.args[0], 1)
    with self.assertRaisesRegex(RedditAPIError, 'wait'):
      client.comments()
    self.assertEqual(urlopen.call_count, 2)

  @patch('functions.reddit_collection.urllib.request.urlopen')
  def test_denied_access_no_retry_or_secret_in_error(self, urlopen):
    urlopen.side_effect = urllib.error.HTTPError('https://oauth.reddit.com', 403,
                                                'secret-server-message', {}, None)
    with self.assertRaises(RedditAPIError) as exc:
      RedditClient('test-agent', 'secret-token').comments()
    self.assertNotIn('secret', str(exc.exception))
    self.assertEqual(urlopen.call_count, 1)

  @patch('functions.reddit_collection.urllib.request.urlopen')
  def test_missing_credentials_fails_without_network(self, urlopen):
    with patch.dict('os.environ', {'REDDIT_USER_AGENT': 'test-agent'}, clear=True):
      with self.assertRaisesRegex(ValueError, 'import-csv'):
        RedditClient.from_env()
    urlopen.assert_not_called()


if __name__ == '__main__':
  unittest.main()
