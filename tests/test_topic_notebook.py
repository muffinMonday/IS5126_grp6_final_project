"""Execute notebook cells on missing data and a small synthetic topic corpus."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

try:
  import nbformat
except ImportError:
  nbformat = None

from functions.topic_data import export_topic_run


ROOT = Path(__file__).resolve().parents[1]
AS_OF = '2026-10-28T06:00:00+00:00'


@unittest.skipIf(nbformat is None, 'nbformat is needed to validate notebook cells.')
class TopicNotebookTests(unittest.TestCase):
  def run_notebook(self, run_dir, modelling=False):
    notebook = nbformat.read(ROOT / 'notebook/2_topic_discovery.ipynb', as_version=4)
    nbformat.validate(notebook)
    namespace = {'display': lambda *args, **kwargs: None}
    with contextlib.redirect_stdout(io.StringIO()), patch('urllib.request.urlopen', side_effect=AssertionError('Network disabled')) as network:
      for cell in notebook.cells:
        if cell.cell_type != 'code':
          continue
        exec(compile(cell.source, f'notebook:{cell.id}', 'exec'), namespace)
        if cell.id == 'topic-setup':
          namespace.update(RUN_DIR=run_dir, RUN_TOPIC_MODEL=modelling,
                           REUSE_SAVED_MODEL=False,
                           SELECTED_K=3 if modelling else None,
                           PCA_COMPONENTS=20, K_VALUES=[2, 3, 4],
                           SAVE_TOPIC_OUTPUTS=modelling)
          namespace['plt'].show = lambda: namespace['plt'].close('all')
      network.assert_not_called()
    return namespace

  def test_missing_corpus_skips_modelling_without_network_or_synthetic_fallback(self):
    with tempfile.TemporaryDirectory() as tmp:
      namespace = self.run_notebook(Path(tmp))
    self.assertTrue(namespace['corpus'].empty)
    self.assertFalse(namespace['features_ready'])
    self.assertFalse(namespace['final_ready'])

  def test_synthetic_pca_profiles_trends_and_zero_vector_export(self):
    rng = np.random.default_rng(42)
    groups = [
      ['layoff', 'severance', 'redundancy', 'downsizing', 'termination'],
      ['interview', 'recruiter', 'resume', 'application', 'hiring'],
      ['salary', 'benefits', 'compensation', 'bonus', 'paycheck'],
    ]
    rows = []
    for group, terms in enumerate(groups):
      for index in range(30):
        text = ' '.join(rng.choice(terms, 12)) + ' ' + ' '.join(
          f'context{j}' for j in rng.choice(100, 8, replace=False))
        rows.append({'comment_id': f'c{group}x{index}', 'text': text,
                     'created_utc': int(pd.Timestamp(f'2026-{index % 3 + 1:02d}-15', tz='UTC').timestamp()),
                     'subreddit': 'jobs', 'link_id': f't3_post{group}'})
    rows.append({'comment_id': 'zero', 'text': 'uniquefirst uniquesecond uniquethird',
                 'created_utc': rows[0]['created_utc'], 'subreddit': 'jobs', 'link_id': 't3_post'})
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / 'synthetic_run'
      export_topic_run(run_dir, rows, 2026, AS_OF, {'method': 'synthetic_test'})
      namespace = self.run_notebook(run_dir, modelling=True)
      self.assertEqual(namespace['Z'].shape, (90, namespace['pca'].n_components_))
      self.assertEqual(namespace['pca'].n_components_, 20)
      self.assertGreater(namespace['pca_variance'].cumulative_variance.iloc[-1], 0)
      self.assertTrue(np.allclose(np.linalg.norm(namespace['Z'], axis=1), 1))
      self.assertTrue(namespace['final_ready'])
      self.assertFalse(namespace['review_binding_verified'])
      self.assertEqual(namespace['focus_ids'], [])
      self.assertTrue(namespace['topic_profiles'].topic_label.str.startswith('Unreviewed').all())
      self.assertEqual(len(namespace['topic_profiles']), 3)
      self.assertTrue(np.allclose(namespace['topic_shares'].sum(axis=1), 1))
      self.assertEqual(namespace['feature_excluded']['comment_id'].tolist(), ['zero'])
      exported = pd.read_csv(namespace['output'] / 'comment_topics.csv')
      self.assertEqual(len(exported), 91)
      self.assertEqual(exported['topic_status'].eq('zero_tfidf').sum(), 1)
      self.assertTrue(exported['comment_id'].is_unique)
      import joblib
      from functions.topics import topic_feature_text
      bundle = joblib.load(namespace['output'] / 'topic_pipeline.joblib')
      self.assertEqual(bundle['parameters']['label_source'], 'unreviewed')
      transformed = bundle['vectorizer'].transform(topic_feature_text(
        namespace['model_df']['text_topic'], bundle['parameters'].get('protected_phrases', [])))
      projected = bundle['pca'].transform(transformed)
      if bundle['normalizer'] is not None:
        projected = bundle['normalizer'].transform(projected)
      self.assertTrue(np.array_equal(bundle['kmeans'].predict(projected), namespace['final_model'].labels_))

  def test_changed_frozen_model_is_rejected_before_deserialization(self):
    notebook = nbformat.read(ROOT / 'notebook/2_topic_discovery.ipynb', as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    namespace = {'display': lambda *args, **kwargs: None}
    with contextlib.redirect_stdout(io.StringIO()):
      exec(cells['topic-setup'].source, namespace)
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp)
      (path / 'topic_pipeline.joblib').write_bytes(b'changed-model')
      namespace.update(SAVED_MODEL_DIR=path, RUN_DIR=path, RUN_TOPIC_MODEL=True,
                       REUSE_SAVED_MODEL=True, corpus=pd.DataFrame({'comment_id': ['a']}))
      with patch('joblib.load', side_effect=AssertionError('Must not deserialize a changed file')) as load:
        with self.assertRaisesRegex(ValueError, 'binding failed: pipeline'):
          exec(cells['topic-reviewed-model'].source, namespace)
        load.assert_not_called()


if __name__ == '__main__':
  unittest.main()
