import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from added_value.active_learning import active_learning, plot_added_value
from added_value.correlation import correlation
from basic_funcs import available_dataset_indices
import recipe


if __name__ == '__main__':
  config.ensure_output_dirs()
  indices_full_dataset = available_dataset_indices('full', required_types=['tf_idf'])
  if not indices_full_dataset:
    raise RuntimeError(
        "No full tf_idf dataset pickles found. Run data_preparation/clean_docs.py and "
        "data_preparation/dataset_split.py first."
    )
  print('Using full datasets:', indices_full_dataset)
  for index in indices_full_dataset:
    active_learning(index, query_strategy=recipe.QUERY_STRATEGY, importance=recipe.SAMPLE_WEIGHT)

  plot_added_value()

  correlation()
