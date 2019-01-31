import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from basic_funcs import load_results, load_dataset, delete_row_csr
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from active_learning_model.query_strategy import choose_query
from baseline_model.class_imbalance import imba_tech
import config, recipe


def assign_importance(y_update, importance):
  """
  This function determines the sample weight in the partial_fit function;
  :param y_update: a list of binary values for the selected samples
  :param importance: None, string or int, indicating how much weight should be assigned to the query documents;
  options are: 1. integers  2. '10_to_positive'; 3. '10_to_negative'; 4. '10_to_both' (same to the '10')
  :return: one-dimensional array or list of integers;
  """

  sample_weight_list = []

  if importance is None:

    return None

  elif str(importance) == '10_to_positive':

    for label in y_update.values:
      if label == 1:
        sample_weight_list.append(10)
      else:
        sample_weight_list.append(1)

  elif str(importance) == '10_to_negative':

    for label in y_update.values:
      if label == 1:
        sample_weight_list.append(1)
      else:
        sample_weight_list.append(10)

  elif str(importance) == '10_to_both':
    for _ in y_update.values:
      sample_weight_list.append(10)

  else:

    for _ in y_update.values:
      sample_weight_list.append(importance)

  return sample_weight_list


def run_importance(partial_set, query_strategy, optimal_alpha, optimal_tol, importance, random_seed):
  """
  Implement the active learning process, considering "sample_weight" in "partial_fit" function
  :param partial_set: int, index of dataset, ranging from 1 to 5. Each dataset consists of five reviews
  :param query_strategy: string, 'least_confidence', 'certainty_positive' or 'certainty_positive_negative'
  :return: Store Yield and Burden for each model
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

  X_train, y_train, X_u_pool, y_u_pool = load_dataset('partial', partial_set, 'tf_idf')

  # Initiate SVM Model
  clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", tol=optimal_tol, alpha=optimal_alpha,
                                   shuffle=True, random_state=1, max_iter=5000)

  # Apply random undersampling
  X_train, y_train = imba_tech(X_train, y_train, 'random_under', random_seed=random_seed)

  # Train the model with the seed
  clf.fit(X_train, y_train)

  # Start Active Learning Loop!
  while len(y_u_pool) != 0:

    predictions = clf.predict(X_u_pool)

    ####################################################################################
    # Calculate Yield and Burden in active learning model
    cm = confusion_matrix(y_u_pool, predictions)

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

    # Get indices of all real true positives
    N_indices = np.nonzero(y_u_pool)[0]

    # Get distance of all relevant documents to the hyperplane
    d_relevant = distance[N_indices]

    # Record measurements
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

    print(" Length of Unlabelled Pool", len(y_u_pool))

    # Remove X and y of query documents from Unlabelled pool
    X_u_pool = delete_row_csr(X_u_pool, query_indices)
    y_u_pool = y_u_pool.drop(y_u_pool.index[query_indices])

    # Update the Classifier by partial fit
    clf = clf.partial_fit(X_update, y_update, classes=[0, 1], sample_weight=assign_importance(y_update, importance))

    # When the unlabelled pool is empty, stop the loop
    if len(y_u_pool) == 0:
      break

    tp_r = len(tp_reviewer_list)

    num_of_iter += 1

  result_obj = {
    'importance': importance,
    'query': number_query_docs,
    'yield': yield_list,
    'burden': burden_list,
  }

  with open('../Consolidation/sample_weight/{}_importance_{}_undersamp_{}.pickle'.
                format(partial_set, importance, random_seed), 'wb') as handle:
    pickle.dump(result_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return result_obj


def test_importance(partial_sets, optimal_alpha, optimal_tol):
  """
  Implement run_importance() function for models using different sample weights, datasets, and seeds.
  :param partial_sets: a list of integers, indicating different datasets;
  :param optimal_alpha: float, the optimal value of alpha
  :param optimal_tol: float, the optimal value of tolerance
  :return:
  """

  importance_list = [1, 2, 4, 6, 8, 10, 20, 30, 40]

  seeds = list(range(5, 11))

  query_strategy = recipe.QUERY_STRATEGY

  for importance in importance_list:

    for partial_set in partial_sets:

      for random_seed in seeds:

        run_importance(partial_set, query_strategy, optimal_alpha, optimal_tol, importance, random_seed)


def plot_importance():
  """
  Make boxplot of results of this experiment
  :return: PDF graph (including boxplots and dots)
  """

  importance_list = [1, 2, 4, 6, 8, 10, 20, 30, 40]

  partial_sets = [1, 2, 3, 4, 5]

  random_seeds = [1, 2, 3, 4, 5]

  compare_importance = []

  for importance in importance_list:

    ilist = []

    for partial_set in partial_sets:

      for random_seed in random_seeds:
        query_al, yield_al, burden_al = load_results(partial_set, recipe.QUERY_STRATEGY,
                                                     importance=importance, random_seed=random_seed)

        # Get 1-burden when yield first stay at 100%
        yield_al = np.array(yield_al)
        index = np.where(yield_al == 1)[0]

        # Add (1-burden) to the list
        ilist.append(1 - burden_al[index[0]])

    compare_importance.append(ilist)

  plt.figure()
  axes = plt.gca()
  axes.set_ylim([-0.1, 1])

  plt.boxplot(compare_importance)


  # Make dot graph with each model representing as one dot in the same graph
  plt.xticks(range(1, len(importance_list) + 1), importance_list)
  plt.ylabel('(1-Burden) at 100% Yield')

  color = ['r', 'g', 'b', 'y', 'c']

  for i in range(len(compare_importance)):
    y = compare_importance[i]
    x = list(np.repeat(i + 1, len(y)))
    plt.scatter(x, y, c='r', alpha=0.6, s=16)

    x = list(np.random.normal(i + 1, 0.04, size=len(y)))

    for j in range(5):
      plt.scatter(x[5 * j: 5 * j + 5], y[5 * j: 5 * j + 5], c=color[j], alpha=0.6, s=16)

  # plt.show()
  plt.savefig(config.DOCUMENT_WEIGHT)


def plot_relevant_importance():

  partial_sets = [1, 2, 3, 4, 5]

  compare_importance = []

  importance_list = [1, '10_to_positive', '10_to_negative', 10]

  # For each tested value for sample weight
  for importance in importance_list:

    ilist = []

    # For each dataset
    for partial_set in partial_sets:

      # For each random seed
      for random_seed in range(0, 5):
        query_al, yield_al, burden_al = load_results(partial_set, recipe.QUERY_STRATEGY,
                                                     importance=importance, random_seed=random_seed)

        # Get 1-burden when yield first stay at 100%
        yield_al = np.array(yield_al)
        index = np.where(yield_al == 1)[0]
        ilist.append(1 - burden_al[index[0]])

    compare_importance.append(ilist)

  plt.figure()
  axes = plt.gca()
  axes.set_ylim([-0.1, 1])

  plt.boxplot(compare_importance)

  plt.xticks(range(1, len(importance_list) + 1), importance_list)
  plt.ylabel('(1-Burden) at 100% Yield')

  color = ['r', 'g', 'b', 'y', 'c']

  for i in range(len(compare_importance)):
    y = compare_importance[i]
    x = list(np.repeat(i + 1, len(y)))
    plt.scatter(x, y, c='r', alpha=0.6, s=16)

    x = list(np.random.normal(i + 1, 0.04, size=len(y)))

    for j in range(5):
      plt.scatter(x[5 * j: 5 * j + 5], y[5 * j: 5 * j + 5], c=color[j], alpha=0.6, s=16)

  # plt.show()
  plt.savefig(config.RELEVANT_DOCUMENT_WEIGHT)