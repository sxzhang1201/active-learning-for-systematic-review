import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from sklearn import linear_model
import numpy as np
import pickle

from basic_funcs import delete_row_csr, load_added_value_results
from active_learning_model.query_strategy import choose_query
from active_learning_model.importance_test import assign_importance
from basic_funcs import load_full_dataset


def active_learning(index, query_strategy, importance=None):
  """
  Implement active learning process for the full dataset (there are twenty datasets because of the leave-one-out)
  :param index: int, ranging from 1 to 20, and indicating which dataset is used.
  :param query_strategy: string;
  :param importance: int or string;
  :return: Stored WSS@95 and WSS@100 values;
  """

  # Initiate objects
  num_of_iter = 0

  number_query_docs = []
  reviewer_list = []

  wss_95_list = []
  wss_100_list = []

  # Load Dataset
  X_train, y_train, X_u_pool, y_u_pool = load_full_dataset(index)

  weight = int((len(y_train) - y_train.sum()) / y_train.sum())

  # Train SVM Model by SGDClassifier
  clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", tol=1e-6, alpha=1e-5,
                                   shuffle=True, random_state=1, max_iter=5000, class_weight={1: weight})

  # Train the model with the initial labelled pool
  clf.fit(X_train, y_train)

  # Variables Needed for Calculating WSS in active learning
  N = len(y_u_pool)

  N_relevant = y_u_pool.sum()
  print('Number of Relevant Documents', N_relevant)

  # Determine which relevant document will be used as the threshold for calculating WSS@95
  n_relevant_95 = int(round(N_relevant * (1 - 0.95)))

  # Start Active Learning Loop!
  while len(y_u_pool) != 0:

    # predict the remaining items in the unlabelled pool
    clf.predict(X_u_pool)

    # Obtain (or update) the distance (including both positive values and negative values) to the hyperplane
    distance = clf.decision_function(X_u_pool)

    # Get indices of all real true positives
    N_indices = np.nonzero(y_u_pool)[0]

    # Get distance of all relevant documents to the hyperplane
    d_relevant = distance[N_indices]

    ### ---------------------------------

    # if num_of_iter in record_point:
    #
    #   # Plot stacked histogram!
    #   plt.hist([d_relevant, distance], bins=30, stacked=True,
    #            density=True, color=['#56B4E9', '#F0E442'], label=['Relevant Docs', 'All Docs'])
    #   axes = plt.gca()
    #   axes.set_xlim([-20, 20])
    #   axes.set_ylim([0, 0.30])
    #
    #
    #   plt.plot(min(d_relevant), 0, 'o', color='b', label='Min')
    #   plt.plot(max(d_relevant), 0, 'o', color='b', label='Max')
    #
    #
    #
    #   plt.legend()
    #   plt.title('Dataset {} - Iteration {}'.format(i - 10, num_of_iter))
    #
    #   plt.axvline(x=min(d_relevant), color='r', linestyle='dashed', label='%.2f' % min(d_relevant))
    #
    #   plt.axvline(x=max(d_relevant), color='r', linestyle='dashed', label='%.2f' % max(d_relevant))
    #
    #   plt.savefig('Figure/Dataset_2_and_7/{}_{}_distance_histogram.pdf'.format(i - 10, num_of_iter))
    #   # plt.show()
    #
    #   plt.close()

    ### ---------------------------------

    # Calculate WSS@95 (two situations)
    if y_u_pool.sum() > n_relevant_95:
      # Determine the threshold
      threshold_dis_95 = np.sort(d_relevant)[n_relevant_95]
      # Calculate the percentage of documents whose distance is lower than the threshold (negative < positive)
      wss_95 = len(distance[distance < threshold_dis_95]) / N - 0.05

    else:
      # If 95% relevant documents are already queried by the reviewer, then the remaining pool is the saved work.
      wss_95 = len(y_u_pool) / N - 0.05

    # Calculate WSS@100 (two situations)
    # If there is no relevant document in the unlabeled pool, the remaining documents are the saved work.
    if y_u_pool.sum() == 0:
      wss_100 = len(y_u_pool) / N

    # Otherwise, it depends on the threshold
    else:
      threshold_dis_100 = np.sort(d_relevant)[0]
      wss_100 = len(distance[distance < threshold_dis_100]) / N

    # Plot Histogram of Distance
    # if num_of_iter % 50 == 0:
    #
    #   # plot histogram for the distance
    #   plt.hist(distance, bins=300, density=True)
    #   axes = plt.gca()
    #   axes.set_ylim([0, 1])
    #
    #   # plot threshold line
    #   plt.axvline(x=d_relevant[np.argmin(distance[N_indices])], color='r',
    #               linestyle='dashed', label='%.2f' % threshold_dis_100)
    #
    #   plt.figtext(.2, .75, 'WSS@100 = %.2f ' % wss_100)
    #   plt.figtext(.2, .8, 'iteration = {}'.format(num_of_iter * 2))
    #
    #   plt.legend()
    #
    #   plt.savefig('Figure/Distance_Distribution/dive_into_2/{}_iteration_{}.pdf'.format((i - 10), num_of_iter * 2))
    #
    #   # plt.show()
    #
    #   plt.close()

    # Storing Data
    number_query_docs.append(num_of_iter * 2)
    wss_95_list.append(wss_95)
    wss_100_list.append(wss_100)

    # Get indices of selected documents (numpy.array type) following query strategies
    query_indices = choose_query(distance, query_strategy)

    # The X and Y of the queried document
    X_update = X_u_pool[query_indices]  # scipy.sparse.csc.csc_matrix
    y_update = y_u_pool.iloc[query_indices]

    # List of Documents that are to be reviewed by the oracle (reviewer)
    reviewer_list = reviewer_list + list(query_indices)

    print(" Length of Unlabelled Pool", len(y_u_pool))

    # Remove X and y of query documents from Unlabelled pool
    X_u_pool = delete_row_csr(X_u_pool, query_indices)
    y_u_pool = y_u_pool.drop(y_u_pool.index[query_indices])

    # Update the Classifier by partial fit
    clf = clf.partial_fit(X_update, y_update, classes=[0, 1],
                          sample_weight=assign_importance(y_update, importance))

    # When the unlabelled pool is empty, quit the loop
    if len(y_u_pool) == 0:
      break

    num_of_iter += 1

  result_obj = {
    'query_docs': number_query_docs,
    'wss@95': wss_95_list,
    'wss@100': wss_100_list
  }

  with open('../Consolidation/added_value/{}_added_value.pickle'.format(index), 'wb') as handle:
    pickle.dump(result_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return result_obj


def plot_added_value():

  # Plot the trends of WSS@100 deducted by the baseline WSS@100 for twenty reviews

  review_ids = ["CD010438", "CD011984", "CD008643", "CD009944", "CD007427", "CD009593", "CD011549",
              "CD011134", "CD008686", "CD011975", "CD009323", "CD009020", "CD011548", "CD010409",
              "CD008054", "CD010771", "CD009591", "CD008691", "CD010632", "CD007394"]

  for index in range(1, 21):
    number_query_docs, _, wss_100_list = load_added_value_results(index)

    # Substract baseline WSS from the active learning WSS
    wss_100_dif = [(wss_100 - wss_100_list[0]) for wss_100 in wss_100_list]

    # Max value
    dif_max = max(wss_100_dif)

    # Percentage of all unlabelled documents are already selected when max value is reached
    query_max = number_query_docs[wss_100_dif.index(dif_max)]

    # Plot trend graph of WSS different against percentage of query documents
    plt.plot(query_max, dif_max, 'o', color='blue')



    # plt.axvline(x=96, color='b', linestyle='dashed', label='x=96%')
    plt.ylabel('WSS@100', fontsize=10)
    plt.xlabel('Number of documents to be queried from the unlabeled pool', fontsize=10)
    plt.figtext(.2, .8, 'Review: {}'.format(review_ids[index - 1]))
    # Set a baseline line

  plt.axhline(y=0, color='r', linestyle='dashed', label='Baseline WSS')
  plt.legend()
  # plt.show()
  plt.savefig('../Figure/Trend_Diff_WSS@100.pdf')


def get_result_table():
  """
  Print the required variables, and copy them into the Excel table.
  (In the future, need to directly export them into a csv file)
  :return:
  """

  for index in range(1, 21):

    number_query_docs, wss_95_list, wss_100_list = load_added_value_results(index)

    index = wss_100_list.index(max(wss_100_list))

    num_docs = number_query_docs[index]/number_query_docs[-1]

    # Number of query documents when max WSS is reached
    print(num_docs)

    # The baseline WSS
    print(wss_95_list[0])
    print(wss_100_list[0])

    # The Max WSS
    print(max(wss_95_list))
    print(max(wss_100_list))