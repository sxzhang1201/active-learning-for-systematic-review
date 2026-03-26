"""imblearn API: fit_sample (old) vs fit_resample (current)."""


def resample(sampler, X, y):
  resample_fn = getattr(sampler, 'fit_resample', None)
  if resample_fn is not None:
    return resample_fn(X, y)
  return sampler.fit_sample(X, y)
