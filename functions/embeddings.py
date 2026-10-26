import os
import tempfile
import urllib.request
import zipfile
import numpy as np

GLOVE_URL = 'https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'


def glove_filename(dim):
  return f'glove.6B.{dim}d.txt'


def download_glove(dim, glove_dir):
  """Download glove.6B.<dim>d.txt (dim: 50, 100, 200 or 300) into glove_dir.

  The zip is ~820 MB, so it goes to a temp folder (outside any synced project
  folder) and only the requested file is extracted. Returns the extracted path."""
  os.makedirs(glove_dir, exist_ok=True)

  with tempfile.TemporaryDirectory() as tmp:
    zip_path = os.path.join(tmp, 'glove.6B.zip')
    print(f'Downloading {GLOVE_URL} ...')
    urllib.request.urlretrieve(GLOVE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
      z.extract(glove_filename(dim), glove_dir)

  return os.path.join(glove_dir, glove_filename(dim))


def build_embedding_matrix(glove_path, word_index, vocab_size, seed=42):
  """Embedding matrix of shape (vocab_size, dim) aligned with a Keras Tokenizer.

  Row i holds the GloVe vector of the word with index i. Words missing from GloVe
  (including the OOV token) get small random vectors so they can still be learned
  when the layer is trainable. Returns (matrix, coverage), where coverage is the
  share of the kept vocabulary found in GloVe."""
  vectors = {}
  with open(glove_path, encoding='utf8') as f:
    for line in f:
      word, *values = line.rstrip().split(' ')
      if word in word_index and word_index[word] < vocab_size:
        vectors[word] = np.asarray(values, dtype='float32')

  dim = len(next(iter(vectors.values())))
  rng = np.random.default_rng(seed)
  matrix = rng.normal(0, 0.1, (vocab_size, dim)).astype('float32')
  matrix[0] = 0  # padding index

  for word, vec in vectors.items():
    matrix[word_index[word]] = vec

  kept = sum(1 for i in word_index.values() if i < vocab_size)
  return matrix, len(vectors) / kept
