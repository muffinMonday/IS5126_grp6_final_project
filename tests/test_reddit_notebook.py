"""Run notebook 4b's Reddit evaluation cells with real metrics and stub inference; never load or train a model.

Notebook 4b scores four models (two logistic regressions, two Bi-LSTMs) on the r/jobs sample. The tests execute its
helper, load, predict, metrics, comparison, precision-curve and error cells, with `predict_proba` replaced by fixed
probabilities. Stub probabilities are test fixtures, not results from the project's models.
"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
  average_precision_score,
  confusion_matrix,
  ConfusionMatrixDisplay,
  precision_recall_curve,
)

from functions.metrics import toxicity_metrics
from functions.preprocessing import cleanup_text


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebook/4b_LSTM_Logistic_Comparison.ipynb'
CELL_MARKERS = {
  'helpers': 'def evaluate(',
  'reddit-load': 'reddit_path = ',
  'reddit-predict': 'reddit_dependencies = ',
  'reddit-metrics': 'reddit_results = ',
  'shift': 'shift = pd.concat(',
  'sweep': 'def precision_by_share(',
  'reddit-errors': 'reddit_predictions = ',
}
RUN_ORDER = ('helpers', 'reddit-load', 'reddit-predict', 'reddit-metrics', 'reddit-errors')
MODEL_NAMES = ['LR (standard)', 'LR (balanced)', 'LSTM (standard)', 'LSTM (balanced)']
MODEL_COLORS = dict(zip(MODEL_NAMES, ['#7f7f7f', '#8172B3', '#55A868', '#4C72B0']))


class InferenceOnlyTokenizer:
  def __init__(self):
    self.word_index = {'<OOV>': 1, 'the': 2, 'job': 3}
    self.texts = None

  def fit_on_texts(self, texts):
    raise AssertionError('Reddit must never be used to fit a tokenizer.')

  def texts_to_sequences(self, texts):
    self.texts = texts
    return [[self.word_index.get(word, 1) for word in text.split()] for text in texts]


class RedditNotebookTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    cls.cells = {}
    for name, marker in CELL_MARKERS.items():
      matches = [''.join(cell['source']) for cell in notebook['cells']
                 if cell['cell_type'] == 'code' and marker in ''.join(cell['source'])]
      if len(matches) != 1:
        raise AssertionError(f'Expected exactly one {name} cell in notebook 4b.')
      cls.cells[name] = compile(matches[0], name, 'exec')

  def setUp(self):
    self.frame = pd.read_csv(ROOT / 'dataset/reddit/test_100_labeled.csv',
                             dtype=str, keep_default_na=False)
    self.actual = self.frame['is_toxic'].astype(int).to_numpy()
    self.probas = {name: np.where(self.actual == 1, 0.9, 0.1) for name in MODEL_NAMES}
    self.tokenizer = InferenceOnlyTokenizer()
    self.predict_proba = Mock(side_effect=lambda name, texts: self.probas[name])
    self.ns = dict(pd=pd, np=np, plt=plt, Path=Path, display=Mock(), cleanup_text=cleanup_text,
                   toxicity_metrics=toxicity_metrics, confusion_matrix=confusion_matrix,
                   ConfusionMatrixDisplay=ConfusionMatrixDisplay, precision_recall_curve=precision_recall_curve,
                   average_precision_score=average_precision_score, PercentFormatter=PercentFormatter,
                   THRESHOLD=0.37, MAX_LENGTH=80, tokenizer=self.tokenizer, model_names=list(MODEL_NAMES),
                   predict_proba=self.predict_proba, model_colors=MODEL_COLORS)

  def run_cell(self, name, frame=None):
    with redirect_stdout(io.StringIO()), patch.object(plt, 'show'):
      if name == 'reddit-load':
        with patch.object(pd, 'read_csv', return_value=self.frame if frame is None else frame):
          exec(self.cells[name], self.ns)
      else:
        exec(self.cells[name], self.ns)

  def run_all(self):
    for name in RUN_ORDER:
      self.run_cell(name)

  def tearDown(self):
    plt.close('all')

  def test_fixed_threshold_unrounded_predictions_and_metric_definitions(self):
    positive = np.flatnonzero(self.actual == 1)[0]
    negative = np.flatnonzero(self.actual == 0)[0]
    self.probas['LR (balanced)'][positive] = 0.36996  # rounding first would incorrectly make this toxic
    self.probas['LR (balanced)'][negative] = 0.37     # equality with the threshold is toxic
    original = self.frame.copy(deep=True)
    vocabulary = self.tokenizer.word_index.copy()
    self.run_all()

    row = self.ns['reddit_results'].loc['LR (balanced)']
    for name in ('precision', 'recall (TPR)', 'macro_f1'):
      self.assertAlmostEqual(row[name], 0.98)
    self.assertAlmostEqual(row['FPR'], 0.02)
    self.assertEqual((row['TP'], row['FP'], row['FN'], row['TN']), (49, 1, 1, 49))
    self.assertAlmostEqual(row['pr_auc'], 49 / 50 + (1 / 50) * (50 / 51))
    for name in MODEL_NAMES:
      if name != 'LR (balanced)':
        other = self.ns['reddit_results'].loc[name]
        self.assertEqual((other['precision'], other['recall (TPR)'], other['macro_f1'], other['FPR']), (1.0, 1.0, 1.0, 0.0))
    self.assertEqual(self.ns['reddit_label_source'], 'hand labels')
    self.assertEqual(self.ns['THRESHOLD'], 0.37)
    self.assertEqual(self.tokenizer.word_index, vocabulary)
    cleaned = [cleanup_text(t.lower()) for t in self.frame['text']]
    self.assertEqual(self.tokenizer.texts, cleaned)
    self.assertEqual([call.args[0] for call in self.predict_proba.call_args_list], MODEL_NAMES)
    for call in self.predict_proba.call_args_list:
      self.assertEqual(list(call.args[1]), cleaned)
    pd.testing.assert_frame_equal(self.frame, original)

  def test_perfect_predictions_allow_an_empty_error_table(self):
    self.run_all()
    self.assertTrue(self.ns['reddit_errors'].empty)
    for name in MODEL_NAMES:
      self.assertTrue((self.ns['reddit_predictions'][f'error_{name}'] == 'ok').all())

  def test_models_are_scored_independently_in_the_error_table(self):
    positive = np.flatnonzero(self.actual == 1)[0]
    negative = np.flatnonzero(self.actual == 0)[0]
    self.probas['LSTM (standard)'][positive] = 0.1   # a missed toxic comment
    self.probas['LR (standard)'][negative] = 0.9     # a false alarm on another comment
    self.run_all()

    errors = self.ns['reddit_errors'].set_index('comment_id')
    self.assertEqual(len(errors), 2)
    self.assertEqual(errors.loc[self.frame.loc[positive, 'comment_id'], 'error_LSTM (standard)'], 'FN')
    self.assertEqual(errors.loc[self.frame.loc[negative, 'comment_id'], 'error_LR (standard)'], 'FP')
    self.assertEqual(set(errors['n_models_wrong']), {1})
    self.assertEqual(self.ns['reddit_results'].loc['LSTM (standard)', 'FN'], 1)
    self.assertEqual(self.ns['reddit_results'].loc['LR (standard)', 'FP'], 1)
    self.assertEqual(self.ns['reddit_results'].loc['LSTM (balanced)', 'FN'], 0)

  def test_invalid_or_unlabelled_sample_stops_before_inference(self):
    changes = [('is_toxic', ''), ('text', '  '), ('subreddit', 'other'),
               ('annotation_status', 'excluded'), ('annotation_status', 'pending'),
               ('comment_id', self.frame.loc[1, 'comment_id']),
               ('text', self.frame.loc[1, 'text'].upper())]
    for column, value in changes:
      changed = self.frame.copy()
      changed.loc[0, column] = value
      with self.subTest(column=column, value=value[:20]), self.assertRaises(ValueError):
        self.run_cell('reddit-load', changed)
    with self.assertRaises(ValueError):
      self.run_cell('reddit-load', self.frame.drop(columns=['is_toxic']))
    self.predict_proba.assert_not_called()

  def test_human_provenance_is_required_and_reported(self):
    self.run_cell('reddit-load')
    self.assertEqual(self.ns['reddit_label_source'], 'hand labels')
    self.assertTrue((self.frame['annotation_status'] == 'hand_labelled').all())
    self.assertTrue((self.frame['annotator'] == 'human').all())
    for column, value in (('annotator', ''), ('annotator', 'AI:Codex'),
                          ('annotation_status', 'ai_labeled')):
      changed = self.frame.copy()
      changed.loc[0, column] = value
      with self.subTest(column=column, value=value), self.assertRaises(ValueError):
        self.run_cell('reddit-load', changed)

  def test_missing_models_do_not_trigger_training_or_downloads(self):
    self.run_cell('reddit-load')
    self.ns.pop('predict_proba')
    with self.assertRaisesRegex(RuntimeError, 'model-loading'):
      self.run_cell('reddit-predict')
    self.predict_proba.assert_not_called()

  def test_invalid_probabilities_are_not_reported_as_scores(self):
    self.run_cell('helpers')
    self.run_cell('reddit-load')
    for values in (np.full(100, np.nan), np.full(100, 1.1), np.zeros(99)):
      self.probas['LSTM (balanced)'] = values
      with self.subTest(shape=values.shape), self.assertRaisesRegex(ValueError, 'probability'):
        self.run_cell('reddit-predict')

  def test_validation_comparison_and_precision_curves_use_the_confusion_counts(self):
    self.probas['LR (balanced)'][np.flatnonzero(self.actual == 1)[:20]] = 0.1  # 20 missed toxic comments
    self.run_all()
    rng = np.random.default_rng(0)
    y_val = (rng.random(2000) < 0.15).astype(int)
    self.ns['val_results'] = self.ns['results_table'](y_val, {name: np.where(y_val == 1, 0.9, 0.1) for name in MODEL_NAMES})

    self.run_cell('shift')
    shift = self.ns['shift'].astype(float)
    row = shift.loc['LR (balanced)']
    self.assertAlmostEqual(row[('r/jobs sample', 'recall (TPR)')], 0.6)
    self.assertAlmostEqual(row[('change', 'recall (TPR)')], 0.6 - 1.0)
    self.assertAlmostEqual(row[('change', 'FPR')], 0.0)

    self.run_cell('sweep')
    # precision = TPR x share / (TPR x share + FPR x (1 - share)) while TPR and FPR stay fixed
    tp, fn, fp, tn = 30, 20, 10, 40
    expected = lambda share: (30 / 50) * share / ((30 / 50) * share + (10 / 50) * (1 - share))
    precision = self.ns['precision_by_share'](tp, fn, fp, tn, [0.05, 0.5, 0.9])
    for share, value in zip([0.05, 0.5, 0.9], precision):
      self.assertAlmostEqual(value, expected(share))
    self.assertAlmostEqual(precision[1], 0.75)
    self.assertEqual(len(plt.gcf().axes), 2)


if __name__ == '__main__':
  unittest.main()
