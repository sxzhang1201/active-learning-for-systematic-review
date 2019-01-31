from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import Libs.file_storage as fs
import numpy as np, basic_funcs, config

class ModelBuilder(object):
  def __init__(self, matrix_collection, review_names, label_list, grid_parameters, run_similarity = False):
    self.matrix_collection = matrix_collection
    self.review_names = review_names
    self.label_list = label_list
    self.grid_parameters = grid_parameters
    self.run_similarity = run_similarity

    if self.run_similarity is True:
      self.cosine_similarity = fs.load_matrix('cosine_sim_matrix')


  def get_similar_review_names(self, test_review_name, size):
    # Get the cosine similarities ordered and remove the 
    # review itself.
    similarities = self.cosine_similarity.loc[[test_review_name]]
    
    similarities = similarities.drop(test_review_name, axis = 1)
    similarities = similarities.sort_values(by = test_review_name, ascending = False, axis = 1)

    similar_reviews = similarities.iloc[:, 0:size]

    return similar_reviews.columns.values.tolist()


  def split_dataset(self, test_review_name):
    training_review_names = self.review_names

    training_review_names.remove(test_review_name)

    X_train, y_train = self.matrix_collection.get_training_set(training_review_names, self.label_list)

    X_test, y_test = self.matrix_collection.get_test_set(test_review_name, self.label_list)

    dataset = {
      'X_train': X_train,
      'y_train': y_train,
      'X_test': X_test,
      'y_test': y_test
    }

    return dataset


  def unpack_confusion_matrix(self, confusion_matrix):
    # If only one element left in the confusion matrix, then ravel() cannot be used, so here divide it into two paths
    if len(confusion_matrix) > 1:
      _, FP, FN, TP = confusion_matrix.ravel()
    else:
      FP = 0
      FN = 0
      TP = 0

    return FP, FN, TP


  def get_yield(self, confusion_matrix):
    FP, FN, TP = self.unpack_confusion_matrix(confusion_matrix)

    return (self.tp_labelled + TP) / (self.tp_labelled + TP + FN)


  def get_burden(self, confusion_matrix, N):
    FP, FN, TP = self.unpack_confusion_matrix(confusion_matrix)

    return ((len(self.reviewer_list) + TP + FP) / N)


  def run_iteration(self, review_name, similarity_size = None):
    svmSGD = SGDClassifier(loss = "hinge", penalty = "l2")

    if self.run_similarity is True:
      similar_review_names = self.get_similar_review_names(review_name, similarity_size)
      X_train, y_train = self.matrix_collection.get_training_set(similar_review_names, self.label_list)
    else:
      X_train, y_train = self.dataset['X_train'], self.dataset['y_train']

    sampler = RandomUnderSampler(return_indices = False)
    undersampled_X_train, undersampled_y_train = sampler.fit_sample(X_train, y_train)

    if len(undersampled_y_train) < config.NUMBER_CV:
      return

    clf = GridSearchCV(svmSGD, self.grid_parameters, cv = config.NUMBER_CV)
    clf.fit(undersampled_X_train, undersampled_y_train)

    clf = clf.best_estimator_

    ######################

    # Initiate variables
    self.tp_labelled = 0

    # Indices of documents that were queried and read by the reviewer
    self.reviewer_list = []

    # Metrics to be recorded for each iteration in active learning
    yield_list = []
    burden_list = []

    X_test = self.dataset['X_test']
    y_test = self.dataset['y_test']

    total_documents = len(y_test)

    while len(y_test) != 0:
      predictions = clf.predict(X_test)

      ####################################################################################
      # Get confusion matrix for the test data
      cm = confusion_matrix(y_test, predictions)      

      # Calculate Yield and Burden
      cur_yield = self.get_yield(cm)
      cur_burden = self.get_burden(cm, total_documents)

      # Record measurements, respectively meaning the number of selected samples, yield, and burden for each iteration
      yield_list.append(cur_yield)
      burden_list.append(cur_burden)
      ###################################################################################

      # Obtain (or update) the distance (including both positive values and negative values) to the hyperplane
      distance = clf.decision_function(X_test)

      # Get indices of selected documents nearest the hyperplane (implement uncertainty sampling)
      # Select top-2 most uncertain
      query_indices = np.argsort(np.absolute(distance))[0:2]
      
      # The X and Y of the queried document
      X_update = X_test[query_indices]  # scipy.sparse.csc.csc_matrix
      y_update = y_test[query_indices]

      # List of Documents that are to be reviewed by the oracle (reviewer)
      self.reviewer_list = self.reviewer_list + list(query_indices)

      # list of "Relevant" Documents that are screened by the oracle (reviewer)
      for query_index in query_indices:
        if y_test[query_index] == 1:
          self.tp_labelled = self.tp_labelled + 1

      # Remove X and y of query documents from Unlabelled pool
      X_test = basic_funcs.delete_row_csr(X_test, query_indices)
      y_test = np.delete(y_test, query_indices)

      # Update the Classifier by partial fit
      clf = clf.partial_fit(X_update, y_update, classes = [0, 1])

    # Create a dictionary to store the results
    result_obj = {
      'yield': yield_list,
      'burden': burden_list
    }

    fs.store_results(review_name, result_obj, self.run_similarity, similarity_size)


  def run_review(self, review_name):
    self.dataset = self.split_dataset(review_name)

    if self.run_similarity is True:
      for i in range(0, len(config.SIMILAR_SIZES)):
        similarity_size = config.SIMILAR_SIZES[i]

        for i in range(0, config.NUM_ITERATIONS):
          self.run_iteration(review_name, similarity_size)    
    else:
      for i in range(0, config.NUM_ITERATIONS):
        self.run_iteration(review_name)