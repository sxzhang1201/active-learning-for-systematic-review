from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import sqlite3 as sq
import basic_funcs
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


def build_indices(review_ids):
  """
  Build a dictionary recording the min and max indices (indicating the position in a list) of documents for each review;
  :return:  Dictionary {'CD010438': (0, 3250), ..., 'review_id': (min_index, max_index)}
  """

  review_indices = {}

  # Load qrel_abs_train txt file
  clef_data = pd.read_csv(config.TRAIN_QREL_LOCATION, sep="\s+", names=['review_id', 'q0', 'pmid', 'included'])

  # Get index of documents for each review
  for review_id in review_ids:
    index = clef_data.index[clef_data['review_id'] == review_id].tolist()

    # Get the range of index for all documents within each review
    review_indices[review_id] = (min(index), max(index) + 1)

  return review_indices


def train_test_split(array, label, review_set, st_type, naming, review_ids):
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
  review_indices = build_indices(review_ids)

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
      X_test = transformer.fit_transform(X_test)

    # Store data sets in a dictionary
    dataset_obj = {
      'X_train': X_train,
      'y_train': y_train,
      'X_test': X_test,
      'y_test': y_test
    }

    # Store the data object as a pickle file in the Dataset file
    with open('../Datasets/{}_{}_{}.pickle'.format(naming, count, st_type), 'wb') as handle:
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
  # Load all documents
  conn = sq.connect(config.DB_FILE)
  documents = pd.read_sql_query('select pubmed_id, review_id, included, title, abstract from article ', conn)

  # Identify unique review IDs
  review_ids = documents['review_id'].unique()

  # Set seed for random sampling
  np.random.seed(2)

  # List of Reviews in the partial data set and full data set
  partial_set = list(np.random.choice(review_ids, 10, replace=False))
  full_set = list(review_ids.copy())

  # Load array (X) and labels (Y) of all documents
  with (open(config.DOC_TERM_MATRIX, "rb")) as openfile:
    X = pickle.load(openfile)

  y = documents['included']

  # Train-test split of the partial dataset
  train_test_split(X, y, partial_set, 'min_max', 'partial', review_ids)
  train_test_split(X, y, partial_set, 'tf_idf', 'partial', review_ids)

  # Train-test split of the full dataset
  train_test_split(X, y, full_set, 'min_max', 'full', review_ids)
  train_test_split(X, y, full_set, 'tf_idf', 'full', review_ids)