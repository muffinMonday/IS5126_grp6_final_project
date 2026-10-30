"""Checks for topic-only exclusions and strictly verified coverage formatting."""

import hashlib
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from functions.topics import (verified_coverage, vectorize_comments, reduce_topics_to_variance,
                              topic_coordinates, original_space_diagnostics, topic_feature_text)


class TopicModellingTests(unittest.TestCase):
  def test_topic_phrase_protection_keeps_domain_concepts_despite_generic_stop_words(self):
    texts = pd.Series(['the job market and remote work', 'job postings work abroad'])
    before = texts.copy()
    transformed = topic_feature_text(texts, ['job market', 'remote work'])
    self.assertIn('job_market', transformed[0])
    self.assertIn('remote_work', transformed[0])
    self.assertEqual(transformed[1], texts.iloc[1])
    self.assertTrue(texts.equals(before))
    from sklearn.feature_extraction.text import TfidfVectorizer
    vocabulary = TfidfVectorizer(stop_words=['job', 'work']).fit(transformed).vocabulary_
    self.assertIn('job_market', vocabulary)
    self.assertIn('remote_work', vocabulary)

  def test_normalization_transform_can_be_reused_and_leaves_pca_values_unchanged(self):
    original = np.array([[3.0, 4.0], [5.0, 12.0]])
    before = original.copy()
    normalized, transformer = topic_coordinates(original, True)
    self.assertTrue(np.allclose(np.linalg.norm(normalized, axis=1), 1))
    self.assertTrue(np.allclose(transformer.transform(original), normalized))
    self.assertTrue(np.array_equal(original, before))
    with self.assertRaisesRegex(ValueError, 'Zero PCA'):
      topic_coordinates(np.zeros((2, 3)), True)

  def test_common_space_diagnostics_distinguish_separated_from_mixed_labels(self):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(['salary bonus', 'salary paycheck', 'salary benefits',
                                  'interview recruiter', 'interview resume', 'interview hiring'])
    indices = np.arange(6)
    separated = original_space_diagnostics(X, indices, [0, 0, 0, 1, 1, 1], vectorizer, top_n=6)
    mixed = original_space_diagnostics(X, indices, [0, 1, 0, 1, 0, 1], vectorizer, top_n=6)
    self.assertGreater(separated['tfidf_cosine_silhouette'], mixed['tfidf_cosine_silhouette'])
    self.assertLess(separated['mean_top_term_jaccard'], mixed['mean_top_term_jaccard'])

  def test_coverage_formatting_only_is_recovered_without_modifying_file(self):
    original = pd.DataFrame({'month': ['2026-01'],
                             'window_start_utc': ['2026-01-01T00:00:00+00:00'],
                             'partial_month': [False], 'source_window_complete': [False],
                             'selected': [1000]})
    expected = hashlib.sha256(original.to_csv(index=False, lineterminator='\r\n').encode('utf-8-sig')).hexdigest()
    changed = original.copy()
    changed['month'] = 'Jan-26'
    changed['partial_month'] = 'FALSE'
    changed['source_window_complete'] = 'FALSE'
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / 'coverage.csv'
      changed.to_csv(path, index=False)
      before = path.read_bytes()
      with warnings.catch_warnings(record=True) as caught:
        result = verified_coverage(path, expected)
      self.assertEqual(result['month'].tolist(), ['2026-01'])
      self.assertTrue(caught)
      self.assertEqual(path.read_bytes(), before)
      changed['selected'] = 999
      changed.to_csv(path, index=False)
      with self.assertRaises(ValueError):
        verified_coverage(path, expected)

  def test_topic_notice_exclusion_retains_ids_and_unrelated_mentions(self):
    corpus = pd.DataFrame({
      'comment_id': ['a', 'b', 'c', 'd'], 'month': ['2026-01'] * 4,
      'text_topic': ['thank you for your submission unfortunately your post was removed',
                     'my old manager removed a job listing from the website',
                     'interview salary job interview salary',
                     'uniqueone uniquetwo uniquethree'],
    })
    result = vectorize_comments(corpus, min_df=1, max_df=1.0, exclude_moderation=True)
    self.assertEqual(result['excluded']['comment_id'].tolist(), ['a'])
    self.assertEqual(result['excluded']['reason'].tolist(), ['moderation_notice'])
    self.assertIn('b', result['model_df']['comment_id'].tolist())
    self.assertEqual(len(result['model_df']) + len(result['excluded']), len(corpus))

  def test_target_pca_matches_full_svd_and_preserves_original_feature_variance(self):
    rng = np.random.default_rng(42)
    original = rng.normal(size=(120, 30)) * np.linspace(2.0, 0.1, 30)
    reference = PCA(n_components=0.90, svd_solver='full').fit(original)
    pca, reduced, variance, _ = reduce_topics_to_variance(csr_matrix(original))
    self.assertEqual(pca.n_components_, reference.n_components_)
    self.assertGreaterEqual(variance.cumulative_variance.iloc[-1], 0.90)
    self.assertLess(variance.cumulative_variance.iloc[-2], 0.90)
    independent = np.var(reduced, axis=0, ddof=1).sum() / np.var(original, axis=0, ddof=1).sum()
    self.assertGreaterEqual(independent, 0.90)
    self.assertTrue(np.allclose(reduced.mean(axis=0), 0, atol=1e-10))
    self.assertTrue(np.allclose(pca.explained_variance_ratio_, reference.explained_variance_ratio_))

  def test_target_pca_resource_and_variance_checks(self):
    rng = np.random.default_rng(7)
    matrix = csr_matrix(rng.normal(size=(20, 10)))
    with self.assertRaisesRegex(ValueError, 'exceeds'):
      reduce_topics_to_variance(matrix, memory_budget_gb=1e-12)
    with self.assertRaisesRegex(ValueError, 'strictly between'):
      reduce_topics_to_variance(matrix, variance_target=1.0)
    with self.assertRaisesRegex(ValueError, 'positive variance'):
      reduce_topics_to_variance(csr_matrix(np.ones((20, 10))))


if __name__ == '__main__':
  unittest.main()
