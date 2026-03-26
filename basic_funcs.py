import pickle
import scipy.sparse
import numpy as np
import os
import re
import config

"""
All functions in this module are frequently used for many experiments. Including:
1. Load dataset (partial dataset and full dataset)
2. Load results (added value)
3. Calculate WSS@95 (a metric indicating the workload reduction brought by a model for automated screening
                     in systematic reviews)
4. Remove one row from the csr sparse matrix (for active learning cycle)
"""


def delete_row_csr(mat, indices):
  """
  Remove the rows denoted by ``indices`` from the CSR sparse matrix ``mat``.
  :param mat: csr sparse matrix
  :param indices:  list of indices
  :return:
  """
  if not isinstance(mat, scipy.sparse.csr_matrix):
    raise ValueError("works only for CSR format -- use .tocsr() first")

  indices = list(indices)

  mask = np.ones(mat.shape[0], dtype=bool)

  mask[indices] = False

  return mat[mask]


def _require_pickle(path, what, hint=None):
  if os.path.isfile(path):
    return
  head = "Missing {what}:\n  {path}\n\n".format(what=what, path=path)
  if hint is not None:
    raise FileNotFoundError(head + hint)
  raise FileNotFoundError(
      head
      + "These files are created by the data-prep step, not shipped with the repo. You need:\n"
      "  1) Database with articles (miner_database.db) and term-document matrix pickle "
      "(see config.DOC_TERM_MATRIX)\n"
      "  2) Build the term matrix: python data_preparation/clean_docs.py\n"
      "  3) Then: python data_preparation/dataset_split.py\n"
      "     (writes Datasets/partial_*_*.pickle and Datasets/full_*_*.pickle)\n"
  )


def _load_split_dataset_pickle(path, what):
  _require_pickle(path, what)
  with open(path, "rb") as openfile:
    data_set = pickle.load(openfile)
  X_train, y_train = data_set['X_train'], data_set['y_train']
  X_test, y_test = data_set['X_test'], data_set['y_test']
  return X_train, y_train, X_test, y_test


def load_dataset(naming, count, st_type):
  path = config.dataset_pickle_path(naming, count, st_type)
  return _load_split_dataset_pickle(path, 'dataset pickle')


def available_dataset_indices(naming, required_types=None):
  """
  Return sorted dataset indices that exist for all required standardization types.
  Example filenames: partial_3_min_max.pickle, full_12_tf_idf.pickle
  """
  required_types = required_types or ['tf_idf']
  sets_per_type = []

  for st_type in required_types:
    index_set = set()
    pattern = re.compile(r'^{}_([0-9]+)_{}\.pickle$'.format(re.escape(naming), re.escape(st_type)))
    if os.path.isdir(config.DATASETS_DIR):
      for filename in os.listdir(config.DATASETS_DIR):
        match = pattern.match(filename)
        if match is not None:
          index_set.add(int(match.group(1)))
    sets_per_type.append(index_set)

  if not sets_per_type:
    return []
  return sorted(set.intersection(*sets_per_type))


def load_full_dataset(index):
  path = config.dataset_pickle_path('full', index, 'tf_idf')
  return _load_split_dataset_pickle(path, 'full dataset pickle')


def load_results(partial_set, query_strategy, random_seed=None, importance=None):
  if importance is None:
    path = os.path.join(
        config.CONSOLIDATION_QUERY_STRATEGY,
        '{}_{}_seed_{}_undersamp.pickle'.format(partial_set, query_strategy, random_seed))
  else:
    path = os.path.join(
        config.CONSOLIDATION_SAMPLE_WEIGHT,
        '{}_importance_{}_undersamp_{}.pickle'.format(partial_set, importance, random_seed))

  _require_pickle(
      path,
      'active learning consolidation pickle',
      hint=(
          'Run active_learning_model/run_active_learning.py to populate '
          'Consolidation/query_strategy and Consolidation/sample_weight.\n'
      ),
  )
  with open(path, "rb") as openfile:
    result_obj = pickle.load(openfile)

  return result_obj['query'], result_obj['yield'], result_obj['burden']


def calculate_wss_95(y_test, d_to_hyper):
  """

  :param y_test:
  :param d_to_hyper:
  :return:
  """

  recall = 0.95

  # Number of all the documents in class '1' in the test set
  number_relevant = y_test.sum()

  # Get indices of all real true positives
  index_positive = np.nonzero(y_test)[0]

  # Get probability array for positives
  d_posi = d_to_hyper[index_positive]

  # calculate the number of relevant documents that can be excluded given the recall
  number_removed = int(round(number_relevant * (1 - recall)))

  threshold_dis = np.sort(d_posi)[number_removed]

  wss_95 = len(d_to_hyper[d_to_hyper < threshold_dis]) / len(y_test) - (1 - recall)

  return wss_95


def load_added_value_results(index):
  path = os.path.join(config.CONSOLIDATION_ADDED_VALUE, '{}_added_value.pickle'.format(index))
  _require_pickle(
      path,
      'added-value result pickle',
      hint='Run added_value/run_added_value.py to populate Consolidation/added_value.\n',
  )
  with open(path, "rb") as openfile:
    result_obj = pickle.load(openfile)

  iteration_list = result_obj['query_docs']
  wss_95_list = result_obj['wss@95']
  wss_100_list = result_obj['wss@100']

  return iteration_list, wss_95_list, wss_100_list
