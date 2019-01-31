import numpy as np

class Feature_matrix(object):
  def __init__(self):
    self.feature_matrix = None
    self.review_indices = []
    self.review_names = []
    self.pubmed_ids = []


  def insert_matrix(self, matrix, review_indices, review_names, pubmed_ids):
    self.feature_matrix = matrix
    self.review_indices = review_indices
    self.review_names = review_names
    self.pubmed_ids = pubmed_ids


  def get_review_indices(self):
    return self.review_indices


  def get_review_names(self):
    return self.review_names


  def get_feature_matrix(self):
    return self.feature_matrix


  def get_training_set(self, review_names, labels):
    mask = np.ones(self.feature_matrix.shape[0], dtype = bool)

    for name in review_names:
      review_idx = self.review_indices[self.review_names.index(name)]

      mask[range(review_idx[0], review_idx[1])] = False

    X_train = self.feature_matrix[~mask, :]
    y_train = np.asarray(labels)[~mask]

    return X_train, y_train


  def get_test_set(self, review_name, labels):
    review_idx = self.review_indices[self.review_names.index(review_name)]

    mask = np.ones(self.feature_matrix.shape[0], dtype = bool)
    mask[range(review_idx[0], review_idx[1])] = False

    X_test = self.feature_matrix[~mask, :]
    y_test = np.asarray(labels)[~mask]

    return X_test, y_test