"""LDA helpers restored from the original Python 3.14 bytecode cache."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer

from .topics import topic_feature_text


def count_matrix(feature_result, texts, protected_phrases=()):
  """Count words using the exact vocabulary and phrase rules of Part 2."""
  counter = CountVectorizer(
    ngram_range=(1, 2), stop_words=feature_result['stop_words'],
    vocabulary=feature_result['vectorizer'].get_feature_names_out(),
  )
  return counter.transform(topic_feature_text(texts, protected_phrases)), counter


def topic_coherence(components, counts, top_n=10):
  """Document-level UMass and NPMI, with absent pairs assigned NPMI -1."""
  present = (counts > 0).astype(np.float64).tocsc()
  doc_freq = np.asarray(present.sum(axis=0)).ravel()
  n_docs = counts.shape[0]
  rows = []
  for weights in components:
    top = weights.argsort()[::-1][:top_n]
    together = (present[:, top].T @ present[:, top]).toarray()
    npmi, umass = [], []
    for i in range(1, top_n):
      for j in range(i):
        umass.append(np.log((together[i, j] + 1) / doc_freq[top[j]]))
        p_ij = together[i, j] / n_docs
        if p_ij > 0:
          p_i, p_j = doc_freq[top[i]] / n_docs, doc_freq[top[j]] / n_docs
          npmi.append(np.log(p_ij / (p_i * p_j)) / -np.log(p_ij))
        else:
          npmi.append(-1.0)
    rows.append({'umass': np.mean(umass), 'npmi': np.mean(npmi)})
  return pd.DataFrame(rows).rename_axis('topic')


def match_topics(reference, other):
  """Match topic rows one-to-one by maximum cosine similarity."""
  a = reference / np.linalg.norm(reference, axis=1, keepdims=True)
  b = other / np.linalg.norm(other, axis=1, keepdims=True)
  cosine = a @ b.T
  ref, matched = linear_sum_assignment(-cosine)
  return pd.DataFrame({'topic': ref, 'matched_topic': matched,
                       'cosine': cosine[ref, matched]})


def soft_monthly_shares(theta, months, weights=None):
  """Aggregate topic probabilities, optionally weighting each document."""
  frame = pd.DataFrame(theta)
  w = pd.Series(1.0 if weights is None else np.asarray(weights, dtype=float), index=frame.index)
  weighted = frame.mul(w, axis=0).groupby(np.asarray(months)).sum()
  return weighted.div(weighted.sum(axis=1), axis=0).rename_axis(index='month', columns='topic')
