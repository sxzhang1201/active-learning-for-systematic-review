import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from basic_funcs import load_dataset, calculate_wss_95
from sklearn import linear_model
import pickle
import numpy as np
import config


def run_tf_idf(optimal_alpha, optimal_tol, partial_sets, test_types):
  """
  :param optimal_alpha: float, the optimal value of alpha
  :param optimal_tol: float, the optimal value of tolerance
  :param partial_sets: a set of datasets
  :param test_types: string, 'min_max' (min max scaling) or 'tf_idf' (TF-IDF) in this project
  :return: List of WSS@95 values for different standardization approaches
  """

  compare_list = []

  for standard_type in test_types:

    wss_list = []

    for partial_set in partial_sets:
      # Load the data
      X_train, y_train, X_test, y_test = load_dataset('partial', partial_set, standard_type)

      # Initiate the function
      clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", tol=optimal_tol, alpha=optimal_alpha,
                                       shuffle=True, max_iter=10000, random_state=1)

      # Train a model
      clf.fit(X_train, y_train)

      # Predict the test set
      clf.predict(X_test)

      # Get a list of distance of all documents to the hyperplane (can be negative)
      d_to_hyper = clf.decision_function(X_test)

      # Calculate WSS@95 values
      wss_95 = calculate_wss_95(y_test, d_to_hyper)

      # Append the WSS@95 result to a list
      wss_list.append(wss_95)

    # Put lists of different standardizaton types together
    compare_list.append(wss_list)

  # Store results of this experiment
  with open(config.TF_IDF_RESULTS, 'wb') as handle:
    pickle.dump(compare_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return compare_list


def plot_tf_idf():
  """
  This function plot the boxplot of WSS@95 for different standardization types.
  The input should be a list of lists.
  :return: Boxplot graph
  """

  with (open(config.TF_IDF_RESULTS, "rb")) as openfile:
    tf_idf = pickle.load(openfile)

  plt.boxplot(tf_idf)

  plt.xticks((1, 2), ('TF \n n = 5', 'TF-IDF \n n = 5'))
  plt.ylabel('WSS@95')
  plt.ylim([-0.05, 0.5])

  for i in range(len(tf_idf)):
    y = tf_idf[i]
    x = list(np.repeat(i + 1, len(y)))

    plt.scatter(x, y, c='r', alpha=0.6, s=16)

  # plt.show()
  plt.savefig(config.TF_IDF_FIGURE)
  plt.close()