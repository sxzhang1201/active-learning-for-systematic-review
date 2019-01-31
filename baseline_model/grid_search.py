import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from basic_funcs import load_dataset, calculate_wss_95
from sklearn import linear_model
import numpy as np
import pickle
import recipe
import config


def grid_search(alphas, tols, partial_set):
  """
  Get WSS@95 values of different alpha and tolerance for one dataset
  :param alphas: a list of numbers as test values for alphas
  :param tols: a list of numbers as test values for tolerance
  :param partial_set: dataset used for analysis
  :return: List of list containing wss@95 values with different alpha and tolerance
  """

  heatmap_list = []

  # Load dataset
  X_train, y_train, X_test, y_test = load_dataset('partial', partial_set, 'min_max')

  for alpha in alphas:

    alpha_list = []

    for tol in tols:

      # Initiate a linear SVM classifier
      clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", tol=tol, alpha=alpha,
                                       shuffle=True, max_iter=10000, random_state=1)
      # Train the classifier
      clf.fit(X_train, y_train)

      # Predict the test set
      clf.predict(X_test)

      # Get the distance of all samples to the hyperplane (can be negative)
      d_to_hyper = clf.decision_function(X_test)

      # Calculate WSS@95 (a metric to evaluate workload reduction)
      wss_95 = calculate_wss_95(y_test, d_to_hyper)

      # Add WSS@95 values of different tolerence to a list
      alpha_list.append(wss_95)

    # Add the list of WSS@95 values with different alphas to a list
    heatmap_list.append(alpha_list)

  return heatmap_list


def heatmap(alphas, tols, heatmap_list):
  """
  Plot heatmap given the input of (output of grid_search() function)
  :param alphas: a list of numbers as test values for alphas
  :param tols: a list of numbers as test values for tolerance
  :param heatmap_list: List of list containing wss@95 values with different alpha and tolerance
  :return: Heatmap plot
  """
  plt.figure(figsize=(8, 6))
  plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
  plt.imshow(heatmap_list, interpolation='nearest')
  plt.xlabel('tolerance')
  plt.ylabel('alpha')
  plt.colorbar()
  plt.yticks(np.arange(len(alphas)), alphas)
  plt.xticks(np.arange(len(tols)), tols)
  plt.title('Grid Search WSS@95')
  # plt.show()
  plt.savefig(config.HEAT_MAP)
  plt.close()



def run_grid_search(alphas, tols, partial_sets):
  """
  This function gets all the average WSS@95 values for all datasets with different alpha and tolerance.
  These obtained values are used for plotting heatmap.
  :param alphas: a list of numbers as test values for alphas
  :param tols: a list of numbers as test values for tolerance
  :param partial_sets: a set of datasets used for analysis
  :return: Heatmap of average WSS@95 for a set of datasets
  """

  heatmap_dict = {}

  # Store the WSS@95 values of each dataset into a dictionary
  for partial_set in partial_sets:
    heatmap = grid_search(alphas, tols, partial_set)

    heatmap_dict[partial_set] = heatmap

  heatmap_list = []

  # The following loop aims to average the WSS@95 values for all datasets
  for i in range(len(alphas)):

    temp_list = []

    for partial_set in range(len(partial_sets)):
      temp_list.append(heatmap_dict[partial_sets[partial_set]][i])

    average = np.average(temp_list, axis=0)

    heatmap_list.append(list(average))

  # Store the average WSS@95 of all combinations for all datasets
  with open(config.GRID_SEARCH, 'wb') as handle:
    pickle.dump(heatmap_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return heatmap_list