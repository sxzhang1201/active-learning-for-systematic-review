import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import config
from added_value.active_learning import active_learning, plot_added_value
from added_value.correlation import correlation
import recipe


if __name__ == '__main__':
  config.ensure_output_dirs()
  indices_full_dataset = list(range(1, 21))
  for index in indices_full_dataset:
    active_learning(index, query_strategy=recipe.QUERY_STRATEGY, importance=recipe.SAMPLE_WEIGHT)

  plot_added_value()

  correlation()
