"""Compare PCA dimensions and projected-vector normalization on a frozen corpus."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from functions.topics import (vectorize_comments, reduce_topics, topic_coordinates,
                              evaluate_clusters, original_space_diagnostics,
                              topic_profiles, initialization_stability)


DOMAIN_STOP_WORDS = '''job jobs work working worked time times year years day days
week weeks company companies way make makes making right better bad person op
thank thanks appreciate sure thing things doing look looking getting told tell
ask feel try new lot come use used using help hope let able point question questions
post comment read understand definitely agree wrong exactly correct pretty simply
obviously yep'''.split()
PROTECTED_PHRASES = ['job market', 'job search', 'job hunt', 'job offer', 'job interview',
                     'new job', 'entry level', 'hiring manager', 'laid off', 'time off',
                     'sick leave', 'sick time', 'sick days', 'paid leave', 'minimum wage',
                     'remote work', 'work life', 'work experience', 'part time', 'full time',
                     'cover letter', 'human resources', 'two weeks', 'notice period']


def compare(baseline, dimensions, ks, output, feature_mode='baseline', n_init=10):
  parameters = json.loads((baseline / 'run_parameters.json').read_text(encoding='utf-8'))
  source = baseline.parent / 'comments_clean.csv'
  if hashlib.sha256(source.read_bytes()).hexdigest() != parameters['input_sha256']:
    raise ValueError('Frozen corpus no longer matches the baseline.')
  output.mkdir(parents=True, exist_ok=False)
  corpus = pd.read_csv(source, keep_default_na=False, dtype={'comment_id': str})
  tfidf = parameters['tfidf']
  domain = feature_mode == 'domain'
  features = vectorize_comments(
    corpus, min_df=tfidf['min_df'], max_df=tfidf['max_df'],
    max_features=tfidf['max_features'],
    extra_stop_words=tfidf['stop_words'] + (DOMAIN_STOP_WORDS if domain else []),
    keep_words=['off'], exclude_moderation=parameters['exclude_moderation_notices'],
    protected_phrases=PROTECTED_PHRASES if domain else [],
  )
  X, frame = features['X'], features['model_df']
  if domain:
    original = vectorize_comments(corpus, extra_stop_words=tfidf['stop_words'], keep_words=['off'],
                                  exclude_moderation=parameters['exclude_moderation_notices'])
    positions = original['model_df'].reset_index().set_index('comment_id')['index']
    common_X = original['X'][positions.loc[frame.comment_id].to_numpy()]
    common_vectorizer = original['vectorizer']
  else:
    common_X, common_vectorizer = X, features['vectorizer']
  pca, projected, variance, seconds = reduce_topics(X, n_components=max(dimensions))
  print(f'PCA {projected.shape}: {seconds:.1f}s', flush=True)
  variance.to_csv(output / 'pca_variance.csv', index=False)
  rows, profiles, examples, models = [], [], [], {}
  for dimension in dimensions:
    for normalized in [False, True]:
      Z, normalizer = topic_coordinates(projected[:, :dimension], normalized)
      scores, candidates, indices = evaluate_clusters(Z, ks=ks, n_init=n_init)
      for record in scores.to_dict('records'):
        k = int(record['k'])
        model = candidates[k]
        name = f'pca{dimension}_' + ('l2' if normalized else 'raw') + f'_k{k}'
        _, profile, example = topic_profiles(X, Z, frame, features['vectorizer'], model)
        record.update(original_space_diagnostics(common_X, indices, model.labels_, common_vectorizer))
        if domain:
          record['feature_space_top_term_jaccard'] = original_space_diagnostics(
            X, indices, model.labels_, features['vectorizer'])['mean_top_term_jaccard']
        record.update(configuration=name, components=dimension, l2_normalize=normalized,
                      retained_variance=float(variance.cumulative_variance.iloc[dimension - 1]))
        rows.append(record)
        profiles.append(profile.assign(configuration=name))
        examples.append(example.assign(configuration=name))
        models[name] = model
      pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False)
      pd.concat(profiles).to_csv(output / 'candidate_profiles.csv', index=False)
      pd.concat(examples).to_csv(output / 'candidate_examples.csv', index=False)
      print(f'Completed {dimension}D normalize={normalized}', flush=True)
  bundle = {'pca': pca, 'projected': projected, 'features': features, 'models': models,
            'parameters': parameters, 'dimensions': dimensions, 'ks': ks,
            'feature_mode': feature_mode, 'protected_phrases': PROTECTED_PHRASES if domain else []}
  joblib.dump(bundle, output / 'comparison_bundle.joblib', compress=3)
  (output / 'comparison_manifest.json').write_text(json.dumps({
    'input_sha256': parameters['input_sha256'], 'baseline': str(baseline),
    'dimensions': dimensions, 'ks': ks, 'seed': 42, 'n_init': n_init,
    'feature_mode': feature_mode, 'protected_phrases': PROTECTED_PHRASES if domain else [],
    'domain_stop_words': DOMAIN_STOP_WORDS if domain else [],
    'silhouette_sample_size': 1500, 'note': 'No month or external series used in selection.',
  }, indent=2), encoding='utf-8')
  print('Saved:', output, flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--dimensions', type=int, nargs='+', default=[50, 100, 200, 400])
  parser.add_argument('--ks', type=int, nargs='+', default=[6, 8, 10, 11, 12, 15])
  parser.add_argument('--feature-mode', choices=['baseline', 'domain'], default='baseline')
  parser.add_argument('--n-init', type=int, default=10)
  args = parser.parse_args()
  started = time.perf_counter()
  compare(args.baseline, args.dimensions, args.ks, args.output, args.feature_mode, args.n_init)
  print(f'Total seconds: {time.perf_counter() - started:.1f}')
