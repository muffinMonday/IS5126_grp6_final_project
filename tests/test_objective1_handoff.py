"""Offline checks for the restored Objective 1 helpers and shared exports."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from functions.lda_topics import count_matrix, match_topics, soft_monthly_shares, topic_coherence
from functions.layoff_mentions import layoff_terms
from functions.topics import vectorize_comments
from scripts.collect_layoff_mentions import rebuild


class Objective1HandoffTests(unittest.TestCase):
  def test_phrase_counts_reuse_tfidf_vocabulary(self):
    corpus = pd.DataFrame({
      'comment_id': ['a', 'b', 'c'], 'month': ['2026-01'] * 3,
      'text_topic': ['job search job search salary', 'job search salary', 'salary interview'],
    })
    features = vectorize_comments(corpus, min_df=1, max_df=1.0,
                                  protected_phrases=['job search'])
    counts, counter = count_matrix(features, corpus['text_topic'], ['job search'])
    np.testing.assert_array_equal(counter.get_feature_names_out(),
                                  features['vectorizer'].get_feature_names_out())
    self.assertEqual(counts[0, counter.vocabulary_['job_search']], 2)

  def test_weighted_months_and_matching(self):
    theta = np.array([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    shares = soft_monthly_shares(theta, ['a', 'a', 'b'], [3, 1, 1])
    np.testing.assert_allclose(shares.loc['a'], [0.65, 0.35])
    matched = match_topics(theta[:2], theta[:2][::-1])
    self.assertEqual(matched['matched_topic'].tolist(), [1, 0])
    np.testing.assert_allclose(matched['cosine'], 1)

  def test_coherence_absent_pair(self):
    scores = topic_coherence(np.array([[2., 1.]]), sparse.csr_matrix([[1, 0], [0, 1]]), top_n=2)
    self.assertEqual(scores.loc[0, 'npmi'], -1)
    self.assertEqual(scores.loc[0, 'umass'], 0)

  def test_keyword_boundaries_and_missing_values(self):
    self.assertEqual(layoff_terms(None), [])
    self.assertEqual(layoff_terms('LAID-OFF after layoffs', ['laid_off', 'layoff']),
                     ['laid_off', 'layoff'])
    self.assertEqual(layoff_terms('playoffs and laid out an offer', ['laid_off', 'layoff']), [])

  def test_frozen_archive_counts(self):
    root = Path(__file__).resolve().parents[1]
    original, rebuilt = rebuild(root / 'dataset/reddit/layoff_mentions')
    pd.testing.assert_frame_equal(original, rebuilt, check_dtype=False)


if __name__ == '__main__':
  unittest.main()
