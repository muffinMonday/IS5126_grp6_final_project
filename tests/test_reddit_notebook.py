"""Run the appended cells with real metrics and stub inference; never train a model.

Stub probabilities are test fixtures, not results from the project's Bi-LSTM.
"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions.preprocessing import cleanup_text


ROOT = Path(__file__).resolve().parents[1]
CELL_IDS = ('reddit-load-data', 'reddit-predict', 'reddit-metrics', 'reddit-errors')


class InferenceOnlyTokenizer:
  def __init__(self):
    self.word_index = {'<OOV>': 1, 'the': 2, 'job': 3}
    self.texts = None
    self.sequences = None

  def fit_on_texts(self, texts):
    raise AssertionError('Reddit must never be used to fit a tokenizer.')

  def texts_to_sequences(self, texts):
    self.texts = texts
    self.sequences = [[self.word_index.get(word, 1) for word in text.split()] for text in texts]
    return self.sequences


class RedditNotebookTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    notebook = json.loads((ROOT / 'notebook/4_LSTM_model.ipynb').read_text())
    cls.cells = {cell['id']: compile(''.join(cell['source']), cell['id'], 'exec')
                 for cell in notebook['cells'] if cell['id'] in CELL_IDS}
    if set(cls.cells) != set(CELL_IDS):
      raise AssertionError('Reddit evaluation cells are missing.')

  def setUp(self):
    self.frame = pd.read_csv(ROOT / 'dataset/reddit/test_100_ai_draft.csv',
                             dtype=str, keep_default_na=False)
    self.actual = self.frame['is_toxic'].astype(int).to_numpy()
    self.proba = np.where(self.actual == 1, 0.9, 0.1)
    self.tokenizer = InferenceOnlyTokenizer()
    self.model = Mock(spec=['predict'])
    self.model.predict.return_value = self.proba.reshape(-1, 1)
    self.wrapper = SimpleNamespace(final_model=self.model, threshold=0.37)
    self.padding = Mock(side_effect=self.pad)
    self.ns = dict(pd=pd, np=np, plt=plt, display=Mock(), best_kf=self.wrapper,
                   tokenizer=self.tokenizer, max_length=80,
                   cleanup_text=cleanup_text, pad_sequences=self.padding)

  @staticmethod
  def pad(sequences, *, maxlen, padding, truncating):
    if padding != 'post' or truncating != 'post':
      raise AssertionError('Inference must use the training padding/truncation settings.')
    array = np.zeros((len(sequences), maxlen), dtype='int32')
    for index, sequence in enumerate(sequences):
      array[index, :min(len(sequence), maxlen)] = sequence[:maxlen]
    return array

  def run_cell(self, cell_id, frame=None):
    with redirect_stdout(io.StringIO()), patch.object(plt, 'show'):
      if cell_id == 'reddit-load-data':
        with patch.object(pd, 'read_csv', return_value=self.frame if frame is None else frame):
          exec(self.cells[cell_id], self.ns)
      else:
        exec(self.cells[cell_id], self.ns)

  def tearDown(self):
    plt.close('all')

  def test_fixed_threshold_unrounded_predictions_and_metric_definitions(self):
    positive = np.flatnonzero(self.actual == 1)[0]
    negative = np.flatnonzero(self.actual == 0)[0]
    self.proba[positive] = 0.36996  # rounding first would incorrectly make this toxic
    self.proba[negative] = 0.37     # equality with the inherited threshold is toxic
    original = self.frame.copy(deep=True)
    vocabulary = self.tokenizer.word_index.copy()
    for cell_id in CELL_IDS:
      self.run_cell(cell_id)

    metrics = self.ns['reddit_metrics'].loc['Reddit']
    for name in ('precision', 'recall', 'macro_f1'):
      self.assertAlmostEqual(metrics[name], 0.98)
    self.assertAlmostEqual(metrics['pr_auc'], 49 / 50 + (1 / 50) * (50 / 51))
    self.assertEqual(metrics['label_source'], 'AI / unreviewed labels')
    self.assertEqual(self.ns['reddit_pred'][positive], 0)
    self.assertEqual(self.ns['reddit_pred'][negative], 1)
    self.assertEqual(set(self.ns['reddit_errors']['error_type']),
                     {'missed toxic (FN)', 'false alarm (FP)'})
    self.assertEqual(len(self.ns['reddit_predictions']), 100)
    self.assertEqual(self.wrapper.threshold, 0.37)
    self.assertEqual(self.tokenizer.word_index, vocabulary)
    self.assertEqual(self.tokenizer.texts, [cleanup_text(t.lower()) for t in self.frame['text']])
    self.model.predict.assert_called_once()
    self.padding.assert_called_once_with(self.tokenizer.sequences, maxlen=80,
                                         padding='post', truncating='post')
    self.assertEqual(self.ns['X_reddit'].shape, (100, 80))
    pd.testing.assert_frame_equal(self.frame, original)

  def test_perfect_predictions_allow_an_empty_error_table(self):
    for cell_id in CELL_IDS:
      self.run_cell(cell_id)
    self.assertTrue(self.ns['reddit_errors'].empty)
    self.assertTrue(self.ns['reddit_predictions']['matches_reference'].all())

  def test_invalid_or_unlabelled_sample_stops_before_inference(self):
    changes = [('is_toxic', ''), ('text', '  '), ('subreddit', 'other'),
               ('annotation_status', 'excluded'), ('comment_id', self.frame.loc[1, 'comment_id']),
               ('text', self.frame.loc[1, 'text'].upper())]
    for column, value in changes:
      changed = self.frame.copy()
      changed.loc[0, column] = value
      with self.subTest(column=column, value=value[:20]), self.assertRaises(ValueError):
        self.run_cell('reddit-load-data', changed)
    with self.assertRaises(ValueError):
      self.run_cell('reddit-load-data', self.frame.drop(columns=['is_toxic']))
    self.model.predict.assert_not_called()

  def test_review_status_does_not_promote_ai_labels_to_human_ground_truth(self):
    changed = self.frame.copy()
    changed['annotation_status'] = 'reviewed'
    self.run_cell('reddit-load-data', changed)
    self.assertFalse(self.ns['reddit_human_reviewed'])
    changed['annotator'] = 'A1'
    self.run_cell('reddit-load-data', changed)
    self.assertTrue(self.ns['reddit_human_reviewed'])

  def test_missing_model_does_not_trigger_training_or_downloads(self):
    self.run_cell('reddit-load-data')
    self.ns.pop('best_kf')
    with self.assertRaisesRegex(RuntimeError, 'best-model sections'):
      self.run_cell('reddit-predict')
    self.model.predict.assert_not_called()

  def test_invalid_probabilities_are_not_reported_as_scores(self):
    self.run_cell('reddit-load-data')
    for values in (np.full((100, 1), np.nan), np.full((100, 1), 1.1), np.zeros((99, 1))):
      self.model.predict.return_value = values
      with self.subTest(shape=values.shape), self.assertRaisesRegex(ValueError, 'probability'):
        self.run_cell('reddit-predict')


if __name__ == '__main__':
  unittest.main()
