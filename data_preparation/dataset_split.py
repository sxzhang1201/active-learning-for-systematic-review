import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import sqlite3 as sq
import numpy as np
import pickle
import config


def MaxAbs_Scaler(X_train, X_test):
  """
  Max min standardization: scale each feature by its maximum absolute value.
  :param X_train: array-like training data;
  :param X_test: array-like test data;
  :return: standardized training data and test data, and the scaler
  """
  scaler = MaxAbsScaler()
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, scaler


def build_indices(review_series):
  """
  Build review_id -> (start, end) index ranges from the *actual DB row order*.
  This keeps split indices aligned with the loaded term-document matrix.
  """

  review_indices = {}
  current_review = None
  start = 0

  for i, review_id in enumerate(review_series):
    if current_review is None:
      current_review = review_id
      start = i
      continue
    if review_id != current_review:
      review_indices[current_review] = (start, i)
      current_review = review_id
      start = i

  if current_review is not None:
    review_indices[current_review] = (start, len(review_series))

  return review_indices


def train_test_split(array, label, review_set, st_type, naming, review_series):
  """
  This function implements the leave-one-out split of a given dataset:
  select one review as the test set and other reviews as the training set.
  :param array: array-like matrix
  :param label: list of binary values (1 or 0)
  :param review_set: list of reviews
  :param st_type: string, either 'tf_idf' or 'min_max', indicating which standardization to be used
  :param naming: string, used for naming of the stored dataset
  :return: Split the training and test dataset, and store them.
  """

  # Build the dictionary recording the indices (indicating the position in a list) of documents for each review;
  review_indices = build_indices(review_series)

  # Initiate 'count' used for naming
  count = 1

  # For each loop, one review as the test set, the others as the training set
  for test_review in review_set:

    train_index = []

    # Indices of positions in 'array' and 'label' for the test documents
    test_range = review_indices[test_review]
    test_index = np.array(list(range(test_range[0], test_range[1])))

    # Indices of positions in 'array' and 'label' for the training documents
    train_reviews = review_set.copy()
    train_reviews.remove(test_review)

    for train_review in train_reviews:
      train_range = review_indices[train_review]
      train_index = train_index + list(range(train_range[0], train_range[1]))
    train_index = np.array(train_index)

    # Train-test split of X (array) and Y (label)
    X_test = array[test_index]
    X_train = array[train_index]

    y_train = label.iloc[train_index]
    y_test = label.iloc[test_index]

    # Standardization
    if st_type == 'min_max':
      X_train, X_test, _ = MaxAbs_Scaler(X_train, X_test)

    elif st_type == 'tf_idf':

      transformer = TfidfTransformer()
      X_train = transformer.fit_transform(X_train)
      X_test = transformer.transform(X_test)

    # Store data sets in a dictionary
    dataset_obj = {
      'X_train': X_train,
      'y_train': y_train,
      'X_test': X_test,
      'y_test': y_test
    }

    path = config.dataset_pickle_path(naming, count, st_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
      pickle.dump(dataset_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    count += 1


def run_train_test_split():
  """
  This function split one dataset into the training set and the test set with different standardization;
  :return: The stored datasets using two different standardization types. Each dataset contains four elements:
  1. a matrix of the training set;
  2. a matrix of the test set,
  3. a list of labels for the training set;
  4. a list of labels for the test set;
  """
  config.ensure_output_dirs()
  # Ensure schema exists (Peewee creates tables on import for a new DB file).
  import Database.db  # noqa: F401

  conn = sq.connect(config.DB_FILE)
  documents = pd.read_sql_query(
      'select pubmed_id, review_id, included, title, abstract from article order by rowid',
      conn,
  )
  conn.close()

  if documents.empty:
    raise RuntimeError(
        "The article table is empty. Populate the database first:\n"
        "  1) python data_preparation/fetch_raw.py\n"
        "  2) python data_preparation/insert_docs.py\n"
        "  3) python data_preparation/clean_docs.py  # writes {}\n"
        "  4) python data_preparation/dataset_split.py".format(config.DOC_TERM_MATRIX)
    )

  # Identify unique review IDs
  review_ids = documents['review_id'].unique()
  n_reviews = len(review_ids)
  if n_reviews < 2:
    raise RuntimeError(
        "Need at least 2 unique review_id values for leave-one-review-out splitting; found {}. "
        "Increase FETCH_RAW_MAX_ARTICLES (for example: 500+), re-run fetch_raw.py and insert_docs.py, "
        "then run clean_docs.py and dataset_split.py again.".format(n_reviews)
    )

  # Set seed for random sampling
  np.random.seed(2)

  # Partial set: up to 10 reviews (thesis design); shrink if DB has fewer reviews.
  n_partial = min(10, n_reviews)
  if n_partial < 10:
    print('Note: only {} unique review_id(s) in DB; partial_set size is {} (not 10).'.format(
        n_reviews, n_partial))

  partial_set = list(np.random.choice(review_ids, n_partial, replace=False))
  full_set = list(review_ids)

  # Load array (X) and labels (Y) of all documents
  with (open(config.DOC_TERM_MATRIX, "rb")) as openfile:
    X = pickle.load(openfile)

  y = documents['included']
  if X.shape[0] != len(documents):
    raise RuntimeError(
        "Mismatch between term-document matrix rows ({}) and DB article rows ({}). "
        "Re-run clean_docs.py to rebuild {} from the current DB.".format(
            X.shape[0], len(documents), config.DOC_TERM_MATRIX
        )
    )

  # Train-test split of the partial dataset
  train_test_split(X, y, partial_set, 'min_max', 'partial', documents['review_id'])
  train_test_split(X, y, partial_set, 'tf_idf', 'partial', documents['review_id'])

  # Train-test split of the full dataset
  train_test_split(X, y, full_set, 'min_max', 'full', documents['review_id'])
  train_test_split(X, y, full_set, 'tf_idf', 'full', documents['review_id'])


if __name__ == '__main__':
  run_train_test_split()