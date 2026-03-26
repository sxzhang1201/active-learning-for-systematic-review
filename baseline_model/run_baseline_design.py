import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import config
from basic_funcs import available_dataset_indices
from baseline_model.tf_idf import run_tf_idf, plot_tf_idf
from baseline_model.class_imbalance import run_class_imbalance, plot_class_imbalance
from baseline_model.grid_search import run_grid_search, heatmap


if __name__ == '__main__':

  config.ensure_output_dirs()
  partial_sets = available_dataset_indices('partial', required_types=['min_max', 'tf_idf'])
  if not partial_sets:
    raise RuntimeError(
        "No partial dataset pickles found. Run data_preparation/clean_docs.py and "
        "data_preparation/dataset_split.py first."
    )
  print('Using partial datasets:', partial_sets)

  alphas = list(10.0 ** -np.arange(1, 8))
  tols = list(10.0 ** -np.arange(2, 9))

  optimal_alpha = 1e-6
  optimal_tol = 1e-3

  # 1. Grid search to fine-tune alpha and tolerance of linear SVM SGDClassifier
  heatmap_list = run_grid_search(alphas, tols, partial_sets)
  heatmap(alphas, tols, heatmap_list)

  # 2. Plot boxplot to compare TF-IDF with Min-Max
  tf_idf = run_tf_idf(optimal_alpha, optimal_tol, partial_sets, ['min_max', 'tf_idf'])
  plot_tf_idf()

  # 3. Plot boxplot to compare different techniques to deal with class imbalance
  imba_list = run_class_imbalance(partial_sets, optimal_alpha, optimal_tol, 10)
  plot_class_imbalance()