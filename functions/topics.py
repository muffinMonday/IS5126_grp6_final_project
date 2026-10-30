"""TF-IDF, true sparse PCA and K-means helpers for Objective 1."""

import hashlib
import io
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer
from scipy import sparse
from threadpoolctl import threadpool_limits


def verified_coverage(path, expected_hash):
  """Accept metadata formatting changes only if canonical bytes match the original."""
  raw = path.read_bytes()
  frame = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
  if hashlib.sha256(raw).hexdigest() == expected_hash:
    return frame
  restored = frame.copy()
  try:
    restored['month'] = pd.to_datetime(restored['window_start_utc'], utc=True).dt.strftime('%Y-%m')
    for column in ['partial_month', 'source_window_complete']:
      values = restored[column].astype(str).str.lower()
      if not values.isin(['true', 'false']).all():
        raise ValueError('Unexpected boolean metadata.')
      restored[column] = values.map({'true': True, 'false': False})
    matched = any(hashlib.sha256(restored.to_csv(index=False, lineterminator=end).encode('utf-8-sig')).hexdigest()
                  == expected_hash for end in ['\n', '\r\n'])
  except (KeyError, ValueError):
    matched = False
  if not matched:
    raise ValueError('monthly_coverage.csv differs beyond verifiable formatting changes.')
  warnings.warn('Coverage formatting changed; original metadata verified and restored in memory. File unchanged.',
                UserWarning, stacklevel=2)
  return restored


def vectorize_comments(corpus, min_df=3, max_df=0.90, max_features=10000,
                       extra_stop_words=(), keep_words=('no', 'not', 'nor', 'off'),
                       exclude_moderation=False, protected_phrases=()):
  """Fit one vocabulary to the retrospective corpus, retaining zero-vector IDs."""
  required = {'comment_id', 'month', 'text_topic'}
  if not required.issubset(corpus.columns) or corpus.empty:
    raise ValueError('Need a nonempty corpus with comment_id, month and text_topic.')
  if not corpus['comment_id'].is_unique:
    raise ValueError('Comment IDs must be unique before vectorization.')
  if not corpus['text_topic'].map(lambda value: isinstance(value, str)).all():
    raise ValueError('text_topic must contain strings, not missing values.')
  input_count = len(corpus)
  text = corpus['text_topic']
  notice = ((text.str.contains('thank you for your submission', regex=False)
             & text.str.contains('removed', regex=False))
            | (text.str.contains('no job listings', regex=False)
               & text.str.contains('self promotion', regex=False)
               & text.str.contains('job boards', regex=False)))
  if not exclude_moderation:
    notice[:] = False
  notices = corpus.loc[notice, ['comment_id', 'month']].copy().assign(reason='moderation_notice')
  corpus = corpus.loc[~notice].copy()
  if corpus.empty:
    raise ValueError('No comments remain after topic-only moderation notice filtering.')
  started = time.perf_counter()
  stop_words = sorted((set(ENGLISH_STOP_WORDS) | set(extra_stop_words)) - set(keep_words))
  vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=min_df, max_df=max_df, max_features=max_features,
    stop_words=stop_words, sublinear_tf=True, norm='l2', dtype=np.float32,
  )
  all_vectors = vectorizer.fit_transform(topic_feature_text(corpus['text_topic'], protected_phrases))
  has_features = np.asarray(all_vectors.getnnz(axis=1)).ravel() > 0
  model_df = corpus.loc[has_features].reset_index(drop=True).copy()
  excluded = corpus.loc[~has_features, ['comment_id', 'month']].copy().assign(reason='zero_tfidf')
  X = all_vectors[has_features]
  if not len(model_df):
    raise ValueError('No comments have TF-IDF features.')
  terms = vectorizer.get_feature_names_out()
  weights = np.asarray(X.mean(axis=0)).ravel()
  top = weights.argsort()[::-1][:25]
  return {'vectorizer': vectorizer, 'X': X, 'model_df': model_df,
          'excluded': pd.concat([notices, excluded], ignore_index=True), 'stop_words': stop_words,
          'top_terms': pd.DataFrame({'term': terms[top], 'mean_tfidf': weights[top]}),
          'diagnostics': {'input_comments': input_count, 'model_comments': len(model_df),
                          'moderation_notices': len(notices),
                          'zero_vectors': int((~has_features).sum()), 'vocabulary_size': X.shape[1],
                          'nonzero_entries': X.nnz, 'density': X.nnz / (X.shape[0] * X.shape[1]),
                          'sparse_megabytes': (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6,
                          'fit_seconds': time.perf_counter() - started}}


def topic_feature_text(texts, protected_phrases=()):
  """Protect domain phrases before topic stop words; never alter the source corpus.

  Reuse this exact transformation before vectorizer.transform on new comments.
  """
  phrases = sorted(set(protected_phrases), key=lambda phrase: (-len(phrase), phrase))
  if not phrases:
    return texts
  if any(not re.fullmatch(r'[a-z]+(?: [a-z]+)+', phrase) for phrase in phrases):
    raise ValueError('Protected phrases must be lowercase alphabetic multiword phrases.')
  pattern = re.compile(r'\b(?:' + '|'.join(re.escape(phrase) for phrase in phrases) + r')\b')
  return [pattern.sub(lambda match: match.group().replace(' ', '_'), text) for text in texts]


def reduce_topics(X, n_components=50, seed=42, threads=2):
  """ARPACK PCA centers implicitly; never densify the full TF-IDF matrix."""
  if not 0 < n_components < min(X.shape):
    raise ValueError(f'PCA components must be < min(samples, features); shape={X.shape}.')
  started = time.perf_counter()
  pca = PCA(n_components=n_components, svd_solver='arpack', random_state=seed, whiten=False)
  with threadpool_limits(limits=threads):
    reduced = pca.fit_transform(X)
  if not np.isfinite(reduced).all():
    raise ValueError('PCA produced non-finite coordinates.')
  variance = pd.DataFrame({'component': np.arange(1, n_components + 1),
                           'explained_variance_ratio': pca.explained_variance_ratio_})
  variance['cumulative_variance'] = variance['explained_variance_ratio'].cumsum()
  return pca, reduced, variance, time.perf_counter() - started


def pca_resource_estimate(X):
  """Conservative planning estimate, not a guarantee of operating-system peak RAM."""
  samples, features = X.shape
  rank = min(samples - 1, features)
  covariance_bytes = features * features * 8
  projected_bytes = max(0, samples * rank * 8)
  component_bytes = max(0, features * rank * 8)
  return {'samples': samples, 'features': features,
          'covariance_gb': covariance_bytes / 1e9,
          'estimated_peak_gb': (6 * covariance_bytes + projected_bytes + component_bytes) / 1e9}


def reduce_topics_to_variance(X, variance_target=0.90, seed=42, threads=2,
                              memory_budget_gb=8.0):
  """True centered PCA with the smallest dimension exceeding the variance target.

  covariance_eigh accepts sparse input but creates a dense feature covariance.
  Float64 limits covariance-conditioning error. No vocabulary or rows are removed.
  """
  if not 0 < variance_target < 1:
    raise ValueError('variance_target must lie strictly between 0 and 1.')
  if len(X.shape) != 2 or min(X.shape) < 2:
    raise ValueError('PCA needs at least two rows and two features.')
  plan = pca_resource_estimate(X)
  if memory_budget_gb <= 0 or plan['estimated_peak_gb'] > memory_budget_gb:
    raise ValueError(f'Estimated peak {plan["estimated_peak_gb"]:.2f} GB exceeds '
                     f'budget {memory_budget_gb:.2f} GB; review resources before running.')
  started = time.perf_counter()
  values = X.astype(np.float64)
  mean = np.asarray(values.mean(axis=0)).ravel()
  squared = values.multiply(values).sum() if sparse.issparse(values) else np.square(values).sum()
  total_variance = (float(squared) - values.shape[0] * float(mean @ mean)) / (values.shape[0] - 1)
  if not np.isfinite(total_variance) or total_variance <= 0:
    raise ValueError('The TF-IDF matrix has no finite positive variance.')
  pca = PCA(n_components=variance_target, svd_solver='covariance_eigh',
            random_state=seed, whiten=False)
  with threadpool_limits(limits=threads):
    reduced = pca.fit_transform(values)
  if not np.isfinite(reduced).all():
    raise ValueError('PCA produced non-finite coordinates.')
  independently_retained = float(np.var(reduced, axis=0, ddof=1).sum() / total_variance)
  if independently_retained + 1e-8 < variance_target:
    raise ValueError('Independent variance verification did not reach the requested target.')
  variance = pd.DataFrame({'component': np.arange(1, pca.n_components_ + 1),
                           'explained_variance_ratio': pca.explained_variance_ratio_})
  variance['cumulative_variance'] = variance['explained_variance_ratio'].cumsum()
  pca.variance_target_ = variance_target
  pca.independently_retained_ratio_ = independently_retained
  pca.resource_estimate_ = plan
  return pca, reduced, variance, time.perf_counter() - started


def evaluate_clusters(Z, ks=range(2, 13), seed=42, sample_size=1500, n_init=10, threads=2):
  """Compare k on identical PCA coordinates and a fixed silhouette subset."""
  if sample_size < 3:
    raise ValueError('Silhouette sample_size must be at least 3.')
  values = sorted(set(ks))
  if not values or any(not isinstance(k, int) or not 2 <= k < len(Z) for k in values):
    raise ValueError('Each k must be an integer with 2 <= k < n_samples.')
  indices = np.sort(np.random.default_rng(seed).choice(len(Z), min(sample_size, len(Z)), replace=False))
  models, rows = {}, []
  with threadpool_limits(limits=threads):
    for k in values:
      started = time.perf_counter()
      model = KMeans(n_clusters=k, n_init=n_init, random_state=seed,
                     max_iter=300, algorithm='lloyd').fit(Z)
      sampled_labels = model.labels_[indices]
      n_labels = len(np.unique(sampled_labels))
      silhouette = silhouette_score(Z[indices], sampled_labels) if 1 < n_labels < len(indices) else np.nan
      sizes = pd.Series(model.labels_).value_counts()
      rows.append({'k': k, 'inertia': model.inertia_, 'silhouette': silhouette,
                   'actual_clusters': len(sizes), 'smallest_cluster': int(sizes.min()),
                   'smallest_cluster_share': float(sizes.min() / len(Z)),
                   'largest_cluster_share': float(sizes.max() / len(Z)),
                   'iterations': model.n_iter_, 'seconds': time.perf_counter() - started})
      models[k] = model
  scores = pd.DataFrame(rows)
  scores['inertia_drop_fraction'] = -scores['inertia'].pct_change()
  return scores, models, indices


def topic_profiles(X, Z, frame, vectorizer, model, labels=None, seed=42, top_n=15):
  """Use original TF-IDF means for names, plus nearest and random examples."""
  if X.shape[0] != len(frame) or len(Z) != len(frame):
    raise ValueError('TF-IDF, PCA rows and comment IDs are not aligned.')
  if len(np.unique(model.labels_)) != model.n_clusters:
    raise ValueError('Some requested clusters are empty; inspect the k diagnostics.')
  labels = labels or {}
  terms = vectorizer.get_feature_names_out()
  global_mean = np.asarray(X.mean(axis=0)).ravel()
  data = frame.copy().assign(cluster=model.labels_)
  profiles, examples = [], []
  for cluster_id in range(model.n_clusters):
    mask = model.labels_ == cluster_id
    positions = np.flatnonzero(mask)
    mean = np.asarray(X[mask].mean(axis=0)).ravel()
    ranked = mean.argsort()[::-1]
    ranked = ranked[mean[ranked] > 0][:top_n]
    distinctive = (mean - global_mean).argsort()[::-1]
    distinctive = distinctive[(mean - global_mean)[distinctive] > 0][:10]
    name = labels.get(cluster_id, f'Unreviewed cluster {cluster_id}')
    profiles.append({'cluster': cluster_id, 'topic_label': name, 'n_comments': int(mask.sum()),
                     'share': float(mask.mean()), 'top_terms': ', '.join(terms[ranked]),
                     'distinctive_terms': ', '.join(terms[distinctive]),
                     'label_reviewed': cluster_id in labels})
    distances = np.linalg.norm(Z[positions] - model.cluster_centers_[cluster_id], axis=1)
    nearest = positions[np.argsort(distances)[:5]]
    sampled = np.random.default_rng(seed + cluster_id).choice(positions, min(5, len(positions)), replace=False)
    columns = [name for name in ['comment_id', 'text', 'month', 'cluster', 'permalink'] if name in data]
    for kind, indices in [('nearest', nearest), ('random', sampled)]:
      examples.append(data.iloc[indices][columns].copy().assign(selection=kind))
  profiles = pd.DataFrame(profiles)
  data['topic_label'] = data['cluster'].map(profiles.set_index('cluster')['topic_label'])
  return data, profiles, pd.concat(examples, ignore_index=True)


def initialization_stability(Z, reference, seeds=(7, 21), threads=2, n_init=10):
  """Compare cluster assignments under alternative seeds; ARI ignores label IDs."""
  rows = []
  with threadpool_limits(limits=threads):
    for seed in seeds:
      candidate = KMeans(n_clusters=reference.n_clusters, n_init=n_init,
                         random_state=seed, max_iter=300).fit(Z)
      rows.append({'seed': seed, 'adjusted_rand_index': adjusted_rand_score(reference.labels_, candidate.labels_),
                   'inertia': candidate.inertia_})
  return pd.DataFrame(rows)


def topic_coordinates(projected, l2_normalize=False):
  """Keep variance measurement before normalization; return a reusable transformer."""
  if not np.isfinite(projected).all() or projected.ndim != 2:
    raise ValueError('Expected a finite two-dimensional PCA projection.')
  if not l2_normalize:
    return projected, None
  if np.any(np.linalg.norm(projected, axis=1) <= np.finfo(projected.dtype).eps):
    raise ValueError('Zero PCA projections cannot be normalized reliably.')
  normalizer = Normalizer(norm='l2')
  return normalizer.fit_transform(projected), normalizer


def original_space_diagnostics(X, indices, labels, vectorizer, top_n=10):
  """Compare representations on the same TF-IDF cosine geometry and term overlap.

  These are internal diagnostics, not ground-truth topic accuracy or coherence.
  """
  sampled_labels = np.asarray(labels)[indices]
  n_labels = len(np.unique(sampled_labels))
  score = (silhouette_score(X[indices], sampled_labels, metric='cosine')
           if 1 < n_labels < len(indices) else np.nan)
  top_sets = []
  terms = vectorizer.get_feature_names_out()
  for cluster in np.unique(labels):
    mean = np.asarray(X[np.asarray(labels) == cluster].mean(axis=0)).ravel()
    ranked = mean.argsort()[::-1]
    top_sets.append(set(terms[ranked[mean[ranked] > 0][:top_n]]))
  overlap = [len(a & b) / len(a | b) for i, a in enumerate(top_sets)
             for b in top_sets[i + 1:] if a | b]
  return {'tfidf_cosine_silhouette': score,
          'mean_top_term_jaccard': float(np.mean(overlap)) if overlap else np.nan}
