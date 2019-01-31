import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from basic_funcs import load_results, load_dataset, delete_row_csr
import numpy as np
import pickle, random
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from baseline_model.class_imbalance import imba_tech
import recipe, config


def choose_query(distance, query_strategy):
  """
  This function implements the "query" step in the active learning cycle.
  Note that the batch size is 2.
  :param distance: a list or numpy array, including the distance of all documents to the SVM hyperplane
  :param query_strategy: string, describing which query strategy to be used. Options are:
  1. 'least_confidence'
  2. 'certainty_positive'
  3. 'certainty_positive_negative'
  :return: a list or numpy array with indices for the "selected" documents
  """

  # 'least_confidence' stands for the query strategy - uncertainty sampling
  if query_strategy == 'least_confidence':
    # Get absolute distance to the hyperplane
    abs_distance = np.absolute(distance)
    # Get indices of two documents whose "absolute" distance are the two smallest
    query_indices = np.argsort(abs_distance)[0:2]

  elif query_strategy == 'certainty_positive':
    # Get indices of two documents whose distance to the hyperplane are the two largest (i.e., the first and the second)
    query_indices = np.argsort(distance)[0:2]

  elif query_strategy == 'certainty_positive_negative':
    # Get indices of one documents whose distance to the hyperplane (distance can be negatives) is the smallest
    # and one documents whose distance is the largest (i.e., the first and last one in the order)
    query_indices = np.argsort(distance)[[0, -1]]

  else:
    # if no query strategy is applied, indices of two random documents are assigned
    query_indices = random.sample(range(1, len(distance)), 2)

  return query_indices


def run_query_strategy(partial_set, query_strategy, optimal_alpha, optimal_tol, imba_type, random_seed=None):
  """
  This function implements the active learning process for one dataset.
  :param partial_set: int, index of dataset, ranging from 1 to 5. Each dataset consists of five reviews
  :param query_strategy: string, 'least_confidence', 'certainty_positive' or 'certainty_positive_negative'
  :param optimal_alpha: float, the optimal value of alpha
  :param optimal_tol: float, the optimal value of tolerance
  :param imba_type: string, indicating class imbalance technique
  :param random_seed: int, the number used for the randomization
  :return:
  """

  # Initiate variables
  tp_r = 0
  num_of_iter = 0

  # Indices of documents that were queried and read by the reviewer
  reviewer_list = []
  tp_reviewer_list = []

  # Metrics to be recorded for each iteration in active learning
  number_query_docs = []
  yield_list = []
  burden_list = []

  X_train, y_train, X_u_pool, y_u_pool = load_dataset('partial', partial_set, recipe.STANDARD_TYPE)

  if imba_type == 'weighting':
    # Initiate SVM Model which enables setting class weight
    weight = int((len(y_train) - y_train.sum()) / y_train.sum())
    clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", tol=optimal_tol, alpha=optimal_alpha,
                                     shuffle=True, random_state=1, max_iter=5000, class_weight={1: weight})

  else:
    # Initiate SVM Model
    clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", tol=optimal_tol, alpha=optimal_alpha,
                                     shuffle=True, random_state=1, max_iter=5000)
    # Re-sample the training data due to the undersampling or oversampling - handle class imbalance
    X_train, y_train = imba_tech(X_train, y_train, imba_type, random_seed)

  # Train the model with the seed
  clf.fit(X_train, y_train)

  # Start Active Learning Loop!
  while len(y_u_pool) != 0:

    predictions = clf.predict(X_u_pool)

    ####################################################################################
    # Get confusion matrix for the test data
    cm = confusion_matrix(y_u_pool, predictions)

    # If only one element left in the confusion matrix, then ravel() cannot be used, so here divide it into two paths
    if len(cm) > 1:
      tn_u, fp_u, fn_u, tp_u = cm.ravel()

    else:
      fp_u = 0
      fn_u = 0
      tp_u = 0

    # Calculate Yield and Burden
    yield_al = (tp_r + tp_u)/(tp_r + tp_u + fn_u)
    burden_al = (len(reviewer_list) + tp_u + fp_u)/(len(reviewer_list) + len(y_u_pool))
    ###################################################################################

    # Obtain (or update) the distance (including both positive values and negative values) to the hyperplane
    distance = clf.decision_function(X_u_pool)

    # Record measurements, respectively meaning the number of selected samples, yield, and burden for each iteration
    number_query_docs.append(num_of_iter * 2)
    yield_list.append(yield_al)
    burden_list.append(burden_al)

    # Get indices of selected documents (numpy.array type) following query strategies
    query_indices = choose_query(distance, query_strategy)

    # The X and Y of the queried document
    X_update = X_u_pool[query_indices]  # scipy.sparse.csc.csc_matrix
    y_update = y_u_pool.iloc[query_indices]

    # List of Documents that are to be reviewed by the oracle (reviewer)
    reviewer_list = reviewer_list + list(query_indices)

    # list of "Relevant" Documents that are screened by the oracle (reviewer)
    for query_index in query_indices:
      if y_u_pool.iloc[query_index] == 1:
        tp_reviewer_list.append(query_index)

    # Remove X and y of query documents from Unlabelled pool
    X_u_pool = delete_row_csr(X_u_pool, query_indices)
    y_u_pool = y_u_pool.drop(y_u_pool.index[query_indices])

    # Update the Classifier by partial fit
    clf = clf.partial_fit(X_update, y_update, classes=[0, 1])

    # When the unlabelled pool is empty, stop the loop
    if len(y_u_pool) == 0:
      break

    tp_r = len(tp_reviewer_list)

    num_of_iter += 1

  # Create a dictionary to store the results
  result_obj = {
    'query_strategy': query_strategy,
    'query': number_query_docs,
    'yield': yield_list,
    'burden': burden_list,
  }

  # Store the result dictionary
  with open('../Consolidation/query_strategy/{}_{}_seed_{}_undersamp.pickle'.format(partial_set, query_strategy,
                                                                                 random_seed), 'wb') as handle:
    pickle.dump(result_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return result_obj


def test_query_strategy(partial_sets, optimal_alpha, optimal_tol):
  """
  Implement the run_query_strategy() function for different datasets with different query strategies.
  Tested strategies includes 'least_confidence', 'certainty_positive', and 'certainty_positive_negative'.
  Each model repeats five times using different random seeds for random undersampling technique.
  :param partial_sets: a list of integers, indicating different datasets;
  :param optimal_alpha: float, the optimal value of alpha
  :param optimal_tol: float, the optimal value of tolerance
  :return: List of results are stored.
  """

  # List of query strategies to be tested
  query_strategies = ['least_confidence', 'certainty_positive', 'certainty_positive_negative']

  # Seed list
  seeds = list(range(1, 6))

  for query_strategy in query_strategies:

    for partial_set in partial_sets:

      for seed in seeds:

        run_query_strategy(partial_set, query_strategy, optimal_alpha, optimal_tol,
                           imba_type=recipe.CLASS_IMBALANCE, random_seed=seed)


def plot_query_strategy():
  """
  This function makes a boxplot to represent the results of WSS@95 for different query strategies
  :return:
  """
  query_strategies = ['least_confidence', 'certainty_positive', 'certainty_positive_negative']

  partial_sets = [1, 2, 3, 4, 5]

  seeds = list(range(1, 6))

  # This list will be used for making a boxplot
  compare_query_strategy = []

  for query_strategy in query_strategies:

    ilist = []

    for partial_set in partial_sets:

      for seed in seeds:
        query_al, yield_al, burden_al = load_results(partial_set, query_strategy, random_seed=seed)

        # Get 1-burden when yield first stay at 100%
        yield_al = np.array(yield_al)
        index = np.where(yield_al == 1)[0]

        # Add (1-burden) to the list
        ilist.append(1 - burden_al[index[0]])

    compare_query_strategy.append(ilist)

  plt.figure()
  axes = plt.gca()
  axes.set_ylim([-0.1, 1])

  plt.boxplot(compare_query_strategy)

  plt.xticks((1, 2, 3), ('uncertainty', 'certainty \n(positive)', 'certainty \n (positive and negative)'))
  plt.ylabel('(1-Burden) at 100% Yield')

  color = ['r', 'g', 'b', 'k', 'c']

  # Plot each model as a point in the same graph
  for i in range(len(compare_query_strategy)):
    y = compare_query_strategy[i]

    x = list(np.random.normal(i + 1, 0.04, size=len(y)))

    for j in range(5):
      plt.scatter(x[5 * j: 5 * j + 5], y[5 * j: 5 * j + 5], c=color[j], alpha=0.6, s=16)

  # plt.show()
  plt.savefig(config.QUERY_STRATEGY_FIGURE)