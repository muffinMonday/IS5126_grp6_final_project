import os
import pickle
import pandas as pd
import tensorflow as tf

from .cv import kfold_wrapper


class bayes_search:
  """Bayesian hyperparameter search that scores every candidate with kfold_wrapper.

  `build_model(**params)` must return a compiled, unfitted Keras model.
  `search_space` maps each parameter name to a skopt dimension, e.g.
    {'lstm_units': Integer(4, 64), 'dropout': Real(0.0, 0.5),
     'lr': Real(1e-4, 1e-2, prior='log-uniform')}

  `objective` is an averaged validation metric from kfold_wrapper.METRICS;
  it is maximised, except for 'loss', which is minimised.
  `kfold_kwargs` is passed on to kfold_wrapper (n_splits, epochs, batch_size, ...).
  The search space may also contain 'epochs' and 'batch_size'; those go to
  kfold_wrapper (overriding kfold_kwargs), all other keys go to build_model.

  `selection` decides which trial counts as "best":
    'best'    the trial with the best mean objective
    'stable'  among trials whose mean is within `stability_tolerance` fold standard
              deviations (of the best trial) of the best mean, the one with the lowest
              standard deviation across folds. Near-ties on a small dataset are mostly
              noise, so this prefers the more consistent configuration.

  `fit()` returns the selected kfold_wrapper. Only that one is kept in memory after the
  search; every trial is recorded in `results` (one row per trial: search params +
  summary()), with the selected one flagged in the `selected` column.
  """

  FIT_PARAMS = ('epochs', 'batch_size')

  def __init__(self, build_model, search_space, X_train, y_train, X_val, y_val,
               n_iter=20, n_initial_points=5, objective='macro_f1',
               kfold_kwargs=None, random_state=42,
               selection='best', stability_tolerance=1.0):
    if selection not in ('best', 'stable'):
      raise ValueError("selection must be 'best' or 'stable'")
    if objective not in kfold_wrapper.METRICS:
      raise ValueError(f'objective must be one of {kfold_wrapper.METRICS}')

    self.build_model = build_model
    self.search_space = search_space
    self.data = (X_train, y_train, X_val, y_val)
    self.n_iter = n_iter
    self.n_initial_points = n_initial_points
    self.objective = objective
    self.kfold_kwargs = {'random_state': random_state, **(kfold_kwargs or {})}
    self.random_state = random_state
    self.selection = selection
    self.stability_tolerance = stability_tolerance

    self.results = None
    self.best_trial = None
    self.best_params = None
    self.best_score = None
    self.best_kfold = None

  def _select_trial(self):
    mean = self.results[f'avg_val_{self.objective}']
    std = self.results[f'std_val_{self.objective}']
    sign = 1 if self.objective == 'loss' else -1  # after this, lower is better

    best = (sign * mean).idxmin()
    if self.selection == 'best':
      return best

    tolerance = self.stability_tolerance * std[best]
    candidates = mean.index[sign * mean <= sign * mean[best] + tolerance]
    return std[candidates].idxmin()

  def fit(self):
    from skopt import Optimizer

    names = list(self.search_space)
    optimizer = Optimizer(
      [self.search_space[n] for n in names],
      n_initial_points=self.n_initial_points,
      random_state=self.random_state,
    )
    sign = 1 if self.objective == 'loss' else -1  # skopt minimises

    rows = []
    trials = {}  # trial -> (params, kfold_wrapper); only the selected one is kept at the end
    running_best = None
    for trial in range(1, self.n_iter + 1):
      point = optimizer.ask()
      # skopt returns numpy scalars; the model builder should get plain python values
      params = {n: (v.item() if hasattr(v, 'item') else v) for n, v in zip(names, point)}

      tf.keras.backend.clear_session()
      # epochs / batch_size belong to training, everything else builds the model
      fit_params = {k: v for k, v in params.items() if k in self.FIT_PARAMS}
      model_params = {k: v for k, v in params.items() if k not in self.FIT_PARAMS}
      kf = kfold_wrapper(self.build_model(**model_params), *self.data,
                         **{**self.kfold_kwargs, **fit_params})
      score = kf.avg_val_metrics[self.objective]
      optimizer.tell(point, sign * score)

      rows.append({'trial': trial, **{f'param_{k}': v for k, v in params.items()}, **kf.summary()})
      trials[trial] = (params, kf)
      if running_best is None or sign * score < sign * running_best:
        running_best = score

      print(f'[{trial}/{self.n_iter}] avg val {self.objective}={score:.4f} '
            f'(best {running_best:.4f}) {params}')

    self.results = pd.DataFrame(rows).set_index('trial')
    self.best_trial = self._select_trial()
    self.best_params, self.best_kfold = trials[self.best_trial]
    self.best_score = self.results.loc[self.best_trial, f'avg_val_{self.objective}']
    self.results['selected'] = self.results.index == self.best_trial

    print(f'Selected trial {self.best_trial} ({self.selection}): '
          f'{self.objective} {self.best_score:.4f} '
          f'+- {self.results.loc[self.best_trial, f"std_val_{self.objective}"]:.4f}')
    return self.best_kfold

  def save(self, path):
    """Save the best kfold_wrapper and the trial log under `path`."""
    os.makedirs(path, exist_ok=True)
    self.best_kfold.save(os.path.join(path, 'best_kfold'))

    state = {
      'results': self.results,
      'best_trial': self.best_trial,
      'selection': self.selection,
      'stability_tolerance': self.stability_tolerance,
      'best_params': self.best_params,
      'best_score': self.best_score,
      'objective': self.objective,
      'n_iter': self.n_iter,
      'n_initial_points': self.n_initial_points,
      'kfold_kwargs': self.kfold_kwargs,
      'random_state': self.random_state,
    }
    with open(os.path.join(path, 'search_state.pkl'), 'wb') as f:
      pickle.dump(state, f)

  @classmethod
  def load(cls, path):
    """Restore a saved search: results, best params and the best kfold_wrapper.
    The model builder, search space and data are not stored, so fit() cannot be re-run on it."""
    with open(os.path.join(path, 'search_state.pkl'), 'rb') as f:
      state = pickle.load(f)

    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    obj.best_kfold = kfold_wrapper.load(os.path.join(path, 'best_kfold'))
    return obj
