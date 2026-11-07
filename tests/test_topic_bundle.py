"""Check that the shared topic bundle contains every offline dependency."""

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'dataset/monthly_Topics/kmeans'


class TopicBundleTests(unittest.TestCase):
  def test_inventory_and_every_payload_hash_match(self):
    manifest = json.loads((BUNDLE / 'bundle_manifest.json').read_text(encoding='utf-8'))
    inventory_path = BUNDLE / 'file_inventory.csv'
    self.assertEqual(hashlib.sha256(inventory_path.read_bytes()).hexdigest(), manifest['inventory_sha256'])
    inventory = pd.read_csv(inventory_path)
    self.assertEqual(set(inventory.path), set(manifest['payload_files']))
    total = 0
    for relative, expected in manifest['payload_files'].items():
      path = (BUNDLE / relative).resolve()
      self.assertTrue(path.is_relative_to(BUNDLE.resolve()))
      raw = path.read_bytes()
      self.assertEqual(len(raw), expected['bytes'], relative)
      self.assertEqual(hashlib.sha256(raw).hexdigest(), expected['sha256'], relative)
      total += len(raw)
    self.assertEqual(total, manifest['payload_bytes'])
    self.assertEqual(len(manifest['payload_files']), 31)

  def test_sampling_and_frozen_run_references_are_complete(self):
    prepared = BUNDLE / 'prepared'
    sampling = json.loads((prepared / 'sampling_manifest.json').read_text(encoding='utf-8'))
    for name, expected in sampling['sha256'].items():
      self.assertEqual(hashlib.sha256((prepared / name).read_bytes()).hexdigest(), expected, name)
    frozen = prepared / 'topics_pca100_l2_20261029T105658930366Z'
    parameters = json.loads((frozen / 'run_parameters.json').read_text(encoding='utf-8'))
    for name, expected in parameters['output_sha256'].items():
      self.assertEqual(hashlib.sha256((frozen / name).read_bytes()).hexdigest(), expected, name)
    manifest = json.loads((BUNDLE / 'bundle_manifest.json').read_text(encoding='utf-8'))
    models = [name for name in manifest['payload_files'] if name.endswith('.joblib')]
    self.assertEqual(models, ['prepared/topics_pca100_l2_20261029T105658930366Z/topic_pipeline.joblib'])
    self.assertEqual(hashlib.sha256((frozen / 'topic_pipeline.joblib').read_bytes()).hexdigest(),
                     manifest['final_pipeline_sha256'])

  def test_corpus_and_assignments_retain_the_original_sample(self):
    prepared = BUNDLE / 'prepared'
    corpus = pd.read_csv(prepared / 'comments_clean.csv', keep_default_na=False, dtype={'comment_id': str})
    coverage = pd.read_csv(prepared / 'monthly_coverage.csv')
    assigned = pd.read_csv(prepared / 'topics_pca100_l2_20261029T105658930366Z/comment_topics.csv',
                           keep_default_na=False, dtype={'comment_id': str})
    self.assertEqual(len(corpus), 10000)
    self.assertTrue(corpus.comment_id.is_unique)
    self.assertTrue(corpus.groupby('month').size().eq(1000).all())
    self.assertTrue(corpus.groupby(['month', 'sampling_stratum']).size().eq(50).all())
    self.assertEqual(set(corpus.comment_id), set(assigned.comment_id))
    self.assertEqual(assigned.topic_status.value_counts().to_dict(),
                     {'assigned': 9770, 'zero_tfidf': 124, 'moderation_notice': 106})
    self.assertEqual(coverage.loc[coverage.partial_month, 'month'].tolist(), ['2026-10'])
    self.assertTrue(coverage.selected.eq(1000).all())


if __name__ == '__main__':
  unittest.main()
