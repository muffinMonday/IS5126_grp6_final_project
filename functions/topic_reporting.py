"""Monthly diagnostics for frozen topic assignments; no model fitting."""

import numpy as np
import pandas as pd


def topic_trend_diagnostics(frame, coverage, early_months, late_months):
  """Compare comment shares, equal-post shares and leave-one-stratum-out ranges.

  These are descriptive sensitivity checks, not population confidence intervals.
  Missing whole months are not added as zero activity.
  """
  required = {'comment_id', 'month', 'cluster'}
  if not required.issubset(frame.columns) or frame.empty or not frame.comment_id.is_unique:
    raise ValueError('Need nonempty uniquely identified assigned comments.')
  if not {'month', 'partial_month'}.issubset(coverage.columns) or coverage.month.duplicated().any():
    raise ValueError('Coverage needs one partial_month flag per month.')
  data = frame.copy()
  if not set(data.month).issubset(set(coverage.month)):
    raise ValueError('Assigned months are missing from coverage metadata.')
  counts = pd.crosstab(data.month, data.cluster).sort_index()
  shares = counts.div(counts.sum(axis=1), axis=0)
  monthly = counts.rename_axis(index='month', columns='cluster').stack().rename('n_topic').reset_index()
  monthly['n_assigned'] = monthly.month.map(counts.sum(axis=1))
  monthly['topic_share'] = monthly.n_topic / monthly.n_assigned
  monthly = monthly.merge(coverage, on='month', how='left', validate='many_to_one')
  has_posts = 'link_id' in data and data.link_id.notna().all() and data.link_id.astype(str).ne('').all()
  has_strata = ('sampling_stratum' in data and data.sampling_stratum.notna().all()
                and data.sampling_stratum.astype(str).ne('').all())
  weighted = shares * np.nan
  post_rows = []
  if has_posts:
    data['post_weight'] = 1 / data.groupby(['month', 'link_id']).comment_id.transform('size')
    weighted = data.pivot_table(index='month', columns='cluster', values='post_weight',
                                aggfunc='sum', fill_value=0).reindex(columns=counts.columns, fill_value=0)
    weighted = weighted.div(weighted.sum(axis=1), axis=0)
    for (month, cluster), group in data.groupby(['month', 'cluster']):
      posts = group.link_id.value_counts()
      post_rows.append({'month': month, 'cluster': cluster, 'unique_posts': len(posts),
                        'top_post_share': float(posts.iloc[0] / len(group))})
  post_long = weighted.rename_axis(index='month', columns='cluster').reset_index().melt(
    id_vars='month', var_name='cluster', value_name='equal_post_share')
  monthly = monthly.merge(post_long, on=['month', 'cluster'], validate='one_to_one')
  if post_rows:
    monthly = monthly.merge(pd.DataFrame(post_rows), on=['month', 'cluster'], how='left', validate='one_to_one')
    monthly['unique_posts'] = monthly.unique_posts.fillna(0).astype(int)
  else:
    monthly['unique_posts'] = np.nan
    monthly['top_post_share'] = np.nan
  ranges = []
  if has_strata:
    for month, group in data.groupby('month'):
      totals = group.groupby('sampling_stratum').size()
      if len(totals) < 2:
        continue
      matrix = pd.crosstab(group.sampling_stratum, group.cluster).reindex(columns=counts.columns, fill_value=0)
      rates = (matrix.sum() - matrix).div(len(group) - totals, axis=0)
      for cluster in counts.columns:
        ranges.append({'month': month, 'cluster': cluster, 'minimum_share': rates[cluster].min(),
                       'maximum_share': rates[cluster].max()})
  if ranges:
    monthly = monthly.merge(pd.DataFrame(ranges), on=['month', 'cluster'], how='left', validate='one_to_one')
  else:
    monthly['minimum_share'] = np.nan
    monthly['maximum_share'] = np.nan
  partial = coverage.partial_month.astype(str).str.lower().eq('true')
  complete = set(coverage.loc[~partial, 'month']) & set(counts.index)
  early_ready = bool(early_months) and set(early_months).issubset(complete)
  late_ready = bool(late_months) and set(late_months).issubset(complete)
  rows = []
  for cluster in counts.columns:
    early = counts.loc[early_months, cluster].sum() / counts.loc[early_months].to_numpy().sum() if early_ready else np.nan
    late = counts.loc[late_months, cluster].sum() / counts.loc[late_months].to_numpy().sum() if late_ready else np.nan
    post_early = weighted.loc[early_months, cluster].mean() if early_ready else np.nan
    post_late = weighted.loc[late_months, cluster].mean() if late_ready else np.nan
    rows.append({'cluster': cluster, 'n_comments': int(counts[cluster].sum()),
                 'first3_share': early, 'last3_share': late, 'change_pp': (late - early) * 100,
                 'equal_post_first3_mean': post_early, 'equal_post_last3_mean': post_late,
                 'equal_post_change_pp': (post_late - post_early) * 100,
                 'complete_months': len(complete), 'both_periods_available': early_ready and late_ready})
  return counts, shares, monthly, pd.DataFrame(rows)


def topic_period_sensitivity(frame, coverage, early_months, late_months, clusters):
  """Recompute quarter changes after deleting units or within-month duplicates.

  Keep all assigned topics in each denominator. Deletion envelopes allow no
  deletion or one deletion independently in each period for sensitivity analysis.
  They are sensitivity ranges, not confidence intervals or model refits.
  """
  if set(early_months) & set(late_months):
    raise ValueError('Sensitivity periods must not overlap.')
  _, _, _, summary = topic_trend_diagnostics(frame, coverage, early_months, late_months)
  if not set(clusters).issubset(set(summary.cluster)):
    raise ValueError('Sensitivity clusters must exist in the assigned corpus.')
  data = frame.copy()

  def available(columns):
    return all(column in data and data[column].notna().all()
               and data[column].astype(str).ne('').all() for column in columns)

  def deletion_range(months, cluster, keys):
    if not available(keys):
      return np.nan, np.nan
    group = data.loc[data.month.isin(months)].copy()
    group['target'] = group.cluster.eq(cluster).astype(int)
    units = group.groupby(keys, observed=True).target.agg(['sum', 'size'])
    denominator = len(group) - units['size']
    valid = denominator.gt(0)
    if not valid.any():
      return np.nan, np.nan
    rates = (group.target.sum() - units.loc[valid, 'sum']) / denominator[valid]
    baseline = group.target.mean()
    return min(float(rates.min()), baseline), max(float(rates.max()), baseline)

  has_text_keys = available(['normalized_text_key'])
  deduplicated = data.drop_duplicates(['month', 'normalized_text_key']) if has_text_keys else None
  indexed = summary.set_index('cluster')
  rows = []
  for cluster in clusters:
    baseline = indexed.loc[cluster]
    row = {'cluster': cluster, **{name: baseline[name] for name in [
        'first3_share', 'last3_share', 'change_pp', 'equal_post_change_pp', 'both_periods_available']}}
    for unit, keys in [('post', ['link_id']), ('stratum', ['month', 'sampling_stratum'])]:
      early = deletion_range(early_months, cluster, keys) if baseline.both_periods_available else (np.nan, np.nan)
      late = deletion_range(late_months, cluster, keys) if baseline.both_periods_available else (np.nan, np.nan)
      row[unit + '_delta_min_pp'] = (late[0] - early[1]) * 100
      row[unit + '_delta_max_pp'] = (late[1] - early[0]) * 100
    row['deduplicated_change_pp'] = np.nan
    if baseline.both_periods_available and has_text_keys:
      early = deduplicated.loc[deduplicated.month.isin(early_months), 'cluster']
      late = deduplicated.loc[deduplicated.month.isin(late_months), 'cluster']
      row['deduplicated_change_pp'] = (late.eq(cluster).mean() - early.eq(cluster).mean()) * 100
    rows.append(row)
  return pd.DataFrame(rows)
