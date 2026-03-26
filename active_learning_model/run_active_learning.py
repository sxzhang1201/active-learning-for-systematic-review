import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import config
from active_learning_model.query_strategy import test_query_strategy, plot_query_strategy
from active_learning_model.importance_test import test_importance, plot_importance, plot_relevant_importance
import recipe


if __name__ == '__main__':
  config.ensure_output_dirs()
  partial_sets = [1, 2, 3, 4, 5]

  optimal_alpha = recipe.OPTIMAL_ALPHA
  optimal_tol = recipe.OPTIMAL_TOLERANCE

  # 1. Compare models with different strategies in terms of (1-Burden) at Yield 100%.
  # Train model and record metrics
  test_query_strategy(partial_sets, optimal_alpha, optimal_tol)  # this step takes 12 min
  plot_query_strategy()

  # 2. Test if assigning more weight to query documents can improve performance of active learning models
  test_importance(partial_sets, optimal_alpha, optimal_tol)
  plot_importance()
  plot_relevant_importance()