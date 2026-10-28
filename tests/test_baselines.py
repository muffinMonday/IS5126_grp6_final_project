"""Small, external-only checks of baseline fitting and reported scores."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from functions.baselines import SOURCES, build_tfidf_logreg, run_baselines
from functions.metrics import toxicity_metrics


def sample_frames():
  train, val = [], []
  for source in sorted(SOURCES):
    for label in [0, 1]:
      for i in range(4):
        text = f'{source} train {label} number{i} ' + ('kind helpful' if label == 0 else 'rude attack')
        train.append({'orig_id': f'train-{label}-{i}', 'source': source,
                      'text_raw': text, 'text': text, 'label': label})
      text = f'{source} validationonly label{label} ' + ('kind helpful' if label == 0 else 'rude attack')
      val.append({'orig_id': f'val-{label}', 'source': source,
                  'text_raw': text, 'text': text, 'label': label})
  return pd.DataFrame(train), pd.DataFrame(val)


class BaselineTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.train, cls.val = sample_frames()
    cls.fits = []
    cls.models = []
    original = build_tfidf_logreg

    def record_build(class_weight=None, random_state=42):
      model = original(class_weight, random_state)
      cls.models.append(model)
      original_fit = model.fit

      def record_fit(texts, labels):
        cls.fits.append(list(texts))
        return original_fit(texts, labels)

      model.fit = record_fit
      return model

    with patch('functions.baselines.build_tfidf_logreg', side_effect=record_build):
      cls.result = run_baselines(cls.train, cls.val)

  def test_only_training_texts_fit_vocabulary(self):
    self.assertEqual(len(self.fits), 6)
    train_texts, val_texts = set(self.train.text), set(self.val.text)
    for fitted, model in zip(self.fits, self.models):
      self.assertTrue(set(fitted) <= train_texts)
      self.assertFalse(set(fitted) & val_texts)
      self.assertNotIn('validationonly', model.named_steps['tfidfvectorizer'].vocabulary_)

  def test_loso_excludes_target_source(self):
    for source, fitted in zip(sorted(SOURCES), self.fits[2:]):
      expected = set(self.train.loc[self.train.source != source, 'text'])
      self.assertEqual(set(fitted), expected)
      score = self.result['loso_metrics'].set_index('held_out_source').loc[source]
      self.assertEqual(set(score['train_sources'].split(',')), SOURCES - {source})
      self.assertEqual(score['n_train'], len(expected))
    rows = self.result['loso_predictions']
    self.assertEqual(len(rows), len(self.val))
    self.assertTrue(rows.source.eq(rows.held_out_source).all())

  def test_result_coverage_and_probability(self):
    scores = self.result['metrics']
    self.assertEqual(set(scores.model), {'majority', 'tfidf_logreg', 'tfidf_logreg_balanced'})
    self.assertEqual(len(scores), 15)
    self.assertEqual(set(scores.group), SOURCES | {'all'})
    self.assertEqual(set(self.result['loso_metrics'].held_out_source), SOURCES)
    rows = self.result['predictions']
    self.assertEqual(len(rows), 3 * len(self.val))
    self.assertTrue(np.isfinite(rows.proba).all())
    self.assertTrue(rows.proba.between(0, 1).all())
    self.assertTrue(rows.predicted_label.eq((rows.proba >= .5).astype(int)).all())
    self.assertTrue(rows.error_type.isin(['TN', 'TP', 'FP', 'FN']).all())
    np.testing.assert_allclose(rows.loc[rows.model == 'majority', 'proba'], self.train.label.mean())

  def test_saved_predictions_reproduce_all_metrics(self):
    rows = self.result['predictions']
    for _, saved in self.result['metrics'].iterrows():
      subset = rows[rows.model == saved['model']]
      if saved['group'] != 'all':
        subset = subset[subset.source == saved['group']]
      recalculated = toxicity_metrics(subset.label, subset.proba, threshold=saved['threshold'])
      for metric, value in recalculated.items():
        self.assertAlmostEqual(saved[metric], value)
    for _, saved in self.result['loso_metrics'].iterrows():
      subset = self.result['loso_predictions']
      subset = subset[subset.held_out_source == saved['held_out_source']]
      for metric, value in toxicity_metrics(subset.label, subset.proba).items():
        self.assertAlmostEqual(saved[metric], value)

  def test_invalid_data_and_threshold_are_rejected(self):
    invalid = self.val.copy()
    invalid.loc[0, 'text'] = self.train.text.iloc[0]
    with self.assertRaisesRegex(ValueError, 'overlap'):
      run_baselines(self.train, invalid)
    invalid = self.val.copy()
    invalid.loc[0, 'orig_id'] = self.train.orig_id.iloc[0]
    with self.assertRaisesRegex(ValueError, 'overlap'):
      run_baselines(self.train, invalid)
    invalid = self.train.copy()
    invalid.loc[0, 'label'] = 2
    with self.assertRaisesRegex(ValueError, 'binary'):
      run_baselines(invalid, self.val)
    with self.assertRaisesRegex(ValueError, 'missing columns'):
      run_baselines(self.train.drop(columns=['text']), self.val)
    with self.assertRaisesRegex(ValueError, 'four external sources'):
      run_baselines(self.train[self.train.source != 'dota'], self.val)
    with self.assertRaisesRegex(ValueError, 'threshold'):
      run_baselines(self.train, self.val, threshold=np.nan)

  def test_invalid_model_probabilities_are_rejected(self):
    class BadModel:
      def fit(self, texts, labels):
        return self

      def predict_proba(self, texts):
        return np.tile([-.1, 1.1], (len(texts), 1))

    with patch('functions.baselines.build_tfidf_logreg', return_value=BadModel()):
      with self.assertRaisesRegex(ValueError, 'probabilities'):
        run_baselines(self.train, self.val)


if __name__ == '__main__':
  unittest.main()
