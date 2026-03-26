import pickle
import scipy.sparse
import numpy as np
import os
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
  Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
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


def _require_pickle(path, what):
  if os.path.isfile(path):
    return
  raise FileNotFoundError(
      "Missing {what}:\n  {path}\n\n"
      "These files are created by the data-prep step, not shipped with the repo. You need:\n"
      "  1) Database with articles (miner_database.db) and term-document matrix pickle "
      "(see config.DOC_TERM_MATRIX)\n"
      "  2) Build the term matrix: python data_preparation/clean_docs.py\n"
      "  3) Then: python data_preparation/dataset_split.py\n"
      "     (writes Datasets/partial_*_*.pickle and Datasets/full_*_*.pickle)\n"
      .format(what=what, path=path)
  )


def load_dataset(naming, count, st_type):
  path = config.dataset_pickle_path(naming, count, st_type)
  _require_pickle(path, 'dataset pickle')
  with open(path, "rb") as openfile:
    data_set = pickle.load(openfile)

  X_train, y_train = data_set['X_train'], data_set['y_train']
  X_test, y_test = data_set['X_test'], data_set['y_test']

  return X_train, y_train, X_test, y_test


def load_full_dataset(index):
  path = config.dataset_pickle_path('full', index, 'tf_idf')
  _require_pickle(path, 'full dataset pickle')
  with open(path, "rb") as openfile:
    data_set = pickle.load(openfile)

  X_train, y_train = data_set['X_train'], data_set['y_train']
  X_test, y_test = data_set['X_test'], data_set['y_test']

  return X_train, y_train, X_test, y_test


def load_results(partial_set, query_strategy, random_seed=None, importance=None):
  if importance is None:
    path = os.path.join(
        config.CONSOLIDATION_QUERY_STRATEGY,
        '{}_{}_seed_{}_undersamp.pickle'.format(partial_set, query_strategy, random_seed))
  else:
    path = os.path.join(
        config.CONSOLIDATION_SAMPLE_WEIGHT,
        '{}_importance_{}_undersamp_{}.pickle'.format(partial_set, importance, random_seed))

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
  with open(path, "rb") as openfile:
    result_obj = pickle.load(openfile)

  iteration_list = result_obj['query_docs']
  wss_95_list = result_obj['wss@95']
  wss_100_list = result_obj['wss@100']

  return iteration_list, wss_95_list, wss_100_list
