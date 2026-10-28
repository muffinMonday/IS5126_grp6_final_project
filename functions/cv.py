import os
import pickle
import warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
  accuracy_score,
  average_precision_score,
  f1_score,
  fbeta_score,
  log_loss,
  precision_score,
  recall_score,
  roc_auc_score,
)


def _compute_metrics(y, proba, threshold):
  pred = (proba >= threshold).astype(int)
  return {
    'loss': log_loss(y, proba, labels=[0, 1]),
    'accuracy': accuracy_score(y, pred),
    'precision': precision_score(y, pred, zero_division=0),
    'recall': recall_score(y, pred, zero_division=0),
    'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
    'pr_auc': average_precision_score(y, proba),
    'roc_auc': roc_auc_score(y, proba),
  }


def _run_task(task):
  """Train one fresh model and score it, in this process or in a worker process.

  It takes plain data only: the model arrives as (class, config, compile config), never as
  an object, so every task starts from newly initialised weights and nothing is shared.
  task['kind'] is 'fold' (fit on X_train, validate on X_eval, return metrics and history) or
  'final' (fit on X_train, predict X_eval, return the weights)."""
  # the nightly Keras build wrongly warns that LSTM / Bidirectional drop the mask (masking works)
  warnings.filterwarnings('ignore', message='.*was passed an input with a mask attached.*')
  if task['threads']:
    # must be set before TensorFlow runs its first op in the process; later calls are ignored
    try:
      tf.config.threading.set_intra_op_parallelism_threads(task['threads'])
    except RuntimeError:
      pass

  tf.keras.utils.set_random_seed(task['seed'])
  model = task['model_class'].from_config(task['config'])
  model.compile_from_config(task['compile_config'])

  # Keras metrics such as F1Score need labels shaped (n, 1), not (n,)
  validation_data = None
  if task['kind'] == 'fold':
    validation_data = (task['X_eval'], task['y_eval'].reshape(-1, 1))
  history = model.fit(
    task['X_train'],
    task['y_train'].reshape(-1, 1),
    epochs=task['epochs'],
    batch_size=task['batch_size'],
    validation_data=validation_data,
    verbose=task['verbose'],
  )
  proba_eval = model.predict(task['X_eval'], verbose=0).ravel()

  out = {'kind': task['kind'], 'proba_eval': proba_eval}
  if task['kind'] == 'fold':
    # score both sets with the final weights and dropout off, so train and
    # validation numbers are directly comparable
    out['history'] = history.history
    proba_train = model.predict(task['X_train'], verbose=0).ravel()
    out['train_metrics'] = _compute_metrics(task['y_train'], proba_train, task['threshold'])
    out['val_metrics'] = _compute_metrics(task['y_eval'], proba_eval, task['threshold'])
  else:
    out['weights'] = model.get_weights()
  tf.keras.backend.clear_session()  # worker processes are reused, so release the model
  return out


class kfold_wrapper:
  """Stratified k-fold cross validation on the training set, followed by a
  final evaluation on a held-out validation set.

  `model` must be a compiled, unfitted Keras model. It is never trained
  directly: every fold (and the final fit) trains a fresh copy with newly
  initialised weights, so no information leaks between runs.

  `n_jobs` > 1 trains the folds and the final model in separate processes at the same
  time (n_jobs=-1: one process per fold plus one for the final fit). Each process uses
  `threads_per_job` TensorFlow threads (default: cpu count // number of processes). The
  results match n_jobs=1 up to floating-point differences. The first call starts the worker
  processes (they import TensorFlow), which takes a few seconds; they are then reused.

  Results (all metrics are computed from predictions at `threshold`):
    cv_results          per-fold train/validation metrics (DataFrame)
    avg_train_metrics   mean of each metric across folds, on the fold-train data
    avg_val_metrics     mean of each metric across folds, on the fold-validation data
    std_val_metrics     standard deviation of the same across folds
    validation_metrics  final model (fit on all of X_train) scored on X_val
    hyperparameters     layer and optimizer settings read from the model
  """

  METRICS = ['loss', 'accuracy', 'precision', 'recall', 'macro_f1', 'pr_auc', 'roc_auc']

  def __init__(self, model, X_train, y_train, X_val, y_val,
               n_splits=5, epochs=5, batch_size=16, threshold=0.5,
               random_state=42, verbose=0, n_jobs=1, threads_per_job=None):
    self.model = model
    self.X_train, self.y_train = np.asarray(X_train), np.asarray(y_train).astype(int)
    self.X_val, self.y_val = np.asarray(X_val), np.asarray(y_val).astype(int)
    self.n_splits = n_splits
    self.epochs = epochs
    self.batch_size = batch_size
    self.threshold = threshold
    self.random_state = random_state
    self.verbose = verbose
    self.n_jobs = n_jobs
    self.threads_per_job = threads_per_job

    # kept so a hyperparameter search can log what produced these scores
    self.hyperparameters = self._extract_hyperparameters(model)

    self.histories = {}
    self.cv_results = None
    self.avg_train_metrics = None
    self.avg_val_metrics = None
    self.std_val_metrics = None
    self.validation_metrics = None
    self.final_model = None
    self.oof_proba = None           # out-of-fold probability for every training row
    self.val_proba = None           # final model's probability for every validation row
    self.tuned_threshold = None
    self.tuned_validation_metrics = None

    self._run_all()

  @staticmethod
  def _extract_hyperparameters(model):
    """Scalar settings of each layer and the optimizer, e.g.
    {'LSTM_1__units': 8, 'LSTM_1__dropout': 0.2, 'optimizer__learning_rate': 0.001}."""
    skip = {'name', 'dtype', 'batch_input_shape', 'build_config',
            'class_name', 'module', 'registered_name', 'backward_layer'}
    scalar = (int, float, str, bool)

    params = {}

    def collect(prefix, config):
      # wrapper layers (e.g. Bidirectional) nest the inner layer's config
      for key, value in config.items():
        if key in skip or 'initializer' in key:
          continue
        if isinstance(value, scalar):
          params[f'{prefix}__{key}'] = value
        elif isinstance(value, dict):
          collect(f'{prefix}__{key}', value)

    for i, layer in enumerate(model.layers):
      collect(f'{type(layer).__name__}_{i}', layer.get_config())

    optimizer = model.optimizer
    if optimizer is not None:
      params['optimizer__name'] = type(optimizer).__name__
      params['optimizer__learning_rate'] = float(optimizer.learning_rate)
    return params

  def _metrics(self, y, proba, threshold=None):
    return _compute_metrics(y, proba, self.threshold if threshold is None else threshold)

  def _run_all(self):
    """Train the k fold models and the final model, then collect all results."""
    spec = {
      'model_class': type(self.model),
      'config': self.model.get_config(),
      'compile_config': self.model.get_compile_config(),
      'epochs': self.epochs, 'batch_size': self.batch_size, 'seed': self.random_state,
      'verbose': self.verbose, 'threshold': self.threshold,
    }

    # the final model fits on the whole training set only; X_val stays unseen until its prediction
    skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
    folds = list(skf.split(self.X_train, self.y_train))
    tasks = [{**spec, 'kind': 'final', 'X_train': self.X_train, 'y_train': self.y_train,
              'X_eval': self.X_val, 'y_eval': self.y_val}]
    tasks += [{**spec, 'kind': 'fold',
               'X_train': self.X_train[tr], 'y_train': self.y_train[tr],
               'X_eval': self.X_train[va], 'y_eval': self.y_train[va]} for tr, va in folds]

    n_workers = len(tasks) if self.n_jobs in (None, -1) else min(self.n_jobs, len(tasks))
    if n_workers > 1:
      threads = self.threads_per_job or max(1, (os.cpu_count() or 1) // n_workers)
      for task in tasks:
        task['threads'] = threads
      # the final fit comes first: it has the most rows, so it should start earliest
      results = Parallel(n_jobs=n_workers, backend='loky')(delayed(_run_task)(t) for t in tasks)
    else:
      for task in tasks:
        task['threads'] = None
      results = [_run_task(t) for t in tasks]
    final, fold_results = results[0], results[1:]

    rows = []
    self.oof_proba = np.zeros(len(self.y_train))
    for fold, ((_, val_idx), result) in enumerate(zip(folds, fold_results), start=1):
      self.histories[fold] = SimpleNamespace(history=result['history'])
      # every training row is predicted once, by a model that did not train on it
      self.oof_proba[val_idx] = result['proba_eval']
      row = {'fold': fold}
      row.update({f'train_{k}': v for k, v in result['train_metrics'].items()})
      row.update({f'val_{k}': v for k, v in result['val_metrics'].items()})
      rows.append(row)

    self.cv_results = pd.DataFrame(rows).set_index('fold')
    self.avg_train_metrics = {k: self.cv_results[f'train_{k}'].mean() for k in self.METRICS}
    self.avg_val_metrics = {k: self.cv_results[f'val_{k}'].mean() for k in self.METRICS}
    self.std_val_metrics = {k: self.cv_results[f'val_{k}'].std() for k in self.METRICS}

    # rebuild the final model here from its weights (call it once on a sample to build it)
    self.final_model = spec['model_class'].from_config(spec['config'])
    self.final_model.compile_from_config(spec['compile_config'])
    self.final_model(self.X_train[:1], training=False)
    self.final_model.set_weights(final['weights'])
    self.val_proba = final['proba_eval']
    self.validation_metrics = self._metrics(self.y_val, self.val_proba)

  def evaluate_at(self, threshold):
    """Validation-set metrics of the final model at another threshold (no re-training)."""
    return self._metrics(self.y_val, self.val_proba, threshold)

  def tune_threshold(self, beta=2.0, max_fpr=None, grid=None, plateau=0.01, apply=True):
    """Choose the decision threshold from the out-of-fold predictions only, so the
    validation set stays untouched, then score it once on the validation set.

    Two rules:
      max_fpr=None  F-beta (beta > 1 weights recall more than precision). The curve is
                    bumpy with few rows, so the threshold is the median of all cutoffs
                    scoring within `plateau` of the best, not the single best one. When
                    the model separates the classes only moderately, the curve is flat
                    and the result can run to the edge of the grid (a warning is raised).
      max_fpr=x     the highest-recall cutoff whose out-of-fold false positive rate
                    (share of non-toxic rows flagged) is at most x. Use it when the cost
                    is moderator workload. The cap is met on the out-of-fold predictions;
                    the validation rate can differ by a few points because of noise.

    With apply=True, `threshold` is set to the tuned value, so later evaluation and
    prediction code uses it; `validation_metrics` keeps the original-threshold numbers
    and `tuned_validation_metrics` holds the tuned ones."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    y, p = self.y_train, self.oof_proba
    negatives = (y == 0).sum()
    fpr = np.array([((p >= t) & (y == 0)).sum() / negatives for t in grid])
    fbeta = np.array([fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0) for t in grid])

    if max_fpr is None:
      tuned = float(np.median(grid[fbeta >= fbeta.max() - plateau]))
      rule = f'F{beta:g}'
      if np.isclose(tuned, grid.min()) or np.isclose(tuned, grid.max()):
        warnings.warn(f'The tuned threshold ({tuned:g}) is at the edge of the search grid: '
                      f'{rule} has no interior optimum here. Consider max_fpr instead.')
    else:
      # recall only falls as the cutoff rises, so the lowest allowed cutoff has the most recall
      allowed = grid[fpr <= max_fpr]
      tuned = float(allowed.min()) if len(allowed) else float(grid.max())
      rule = f'max recall with out-of-fold FPR <= {max_fpr:g}'
    tuned = round(tuned, 4)

    self.tuned_threshold = tuned
    self.tuned_validation_metrics = self.evaluate_at(tuned)
    if apply:
      self.threshold = tuned

    pred = (p >= tuned).astype(int)
    return {
      'threshold': tuned,
      'rule': rule,
      'oof_recall': float(recall_score(y, pred, zero_division=0)),
      'oof_precision': float(precision_score(y, pred, zero_division=0)),
      'oof_fpr': float(((pred == 1) & (y == 0)).sum() / negatives),
      'grid': grid,
      'oof_fbeta': fbeta,
      'oof_fpr_curve': fpr,
    }

  def threshold_table(self, thresholds=(0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05)):
    """Recall, precision, false positive rate (FPR) and share of comments flagged at several
    cutoffs. `oof_*` columns use the out-of-fold predictions of the training set (the basis
    for choosing a cutoff); `val_*` columns use the validation set and are for checking only.
    The tuned threshold, if there is one, is added as a row."""
    cutoffs = {round(float(t), 4) for t in thresholds}
    if self.tuned_threshold is not None:
      cutoffs.add(round(self.tuned_threshold, 4))

    rows = []
    for t in sorted(cutoffs, reverse=True):
      row = {'threshold': t}
      for prefix, y, p in (('oof', self.y_train, self.oof_proba), ('val', self.y_val, self.val_proba)):
        pred = (p >= t).astype(int)
        row[f'{prefix}_recall'] = recall_score(y, pred, zero_division=0)
        row[f'{prefix}_precision'] = precision_score(y, pred, zero_division=0)
        row[f'{prefix}_fpr'] = ((pred == 1) & (y == 0)).sum() / (y == 0).sum()
        row[f'{prefix}_flagged'] = pred.mean()
      row['note'] = 'tuned' if t == self.tuned_threshold else ''
      rows.append(row)
    return pd.DataFrame(rows).set_index('threshold')

  def summary(self):
    """Single flat dict: hyperparameters plus all averaged / final metrics."""
    out = dict(self.hyperparameters)
    out.update({f'avg_train_{k}': v for k, v in self.avg_train_metrics.items()})
    out.update({f'avg_val_{k}': v for k, v in self.avg_val_metrics.items()})
    out.update({f'std_val_{k}': v for k, v in self.std_val_metrics.items()})
    out.update({f'holdout_{k}': v for k, v in self.validation_metrics.items()})
    return out

  # attributes that are not saved: the unfitted model, the feature matrices, and objects rebuilt on load
  # (y_train / y_val are kept: they are small and let tune_threshold / evaluate_at work after load)
  _NOT_SAVED = ('model', 'final_model', 'histories', 'X_train', 'X_val')

  def save(self, path):
    """Save final_model (.keras) and everything else needed to inspect the results.
    The training / validation data and the unfitted model are not saved."""
    os.makedirs(path, exist_ok=True)
    self.final_model.save(os.path.join(path, 'final_model.keras'))

    state = {k: v for k, v in self.__dict__.items() if k not in self._NOT_SAVED}
    state['histories'] = {fold: h.history for fold, h in self.histories.items()}
    with open(os.path.join(path, 'state.pkl'), 'wb') as f:
      pickle.dump(state, f)

  @classmethod
  def load(cls, path):
    """Rebuild a kfold_wrapper saved with save(), without re-training anything."""
    with open(os.path.join(path, 'state.pkl'), 'rb') as f:
      state = pickle.load(f)

    obj = cls.__new__(cls)
    histories = state.pop('histories')
    obj.__dict__.update(state)
    # same access pattern as keras History objects: histories[fold].history[metric]
    obj.histories = {fold: SimpleNamespace(history=h) for fold, h in histories.items()}
    obj.final_model = tf.keras.models.load_model(os.path.join(path, 'final_model.keras'))
    return obj
