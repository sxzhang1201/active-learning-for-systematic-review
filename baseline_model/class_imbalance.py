import pickle
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from basic_funcs import load_dataset, calculate_wss_95
from sklearn import linear_model
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import config


def imba_tech(X_train, y_train, imba_type, random_seed):
  """
  Apply undersampling and oversampling to handle class imbalance
  :param X_train: Array-like, the document-term matrix
  :param y_train: list of binary labels (1 or 0)
  :param imba_type: string, according to the type of technique used.
  :param random_seed: int, seed for the randomization
  :return: re-sampled data (including one array-like matrix and one list) or the same data (for 'weighting' imba_type)
  """
  if imba_type == 'random_under':
    # Initiate a sampler for undesampling
    rus = RandomUnderSampler(return_indices=False, random_state=random_seed)
    X_train, y_train = rus.fit_sample(X_train, y_train)

  elif imba_type == 'random_over':
    # Initiate a sampler for oversampling
    ros = RandomOverSampler(random_state=random_seed)
    X_train, y_train = ros.fit_sample(X_train, y_train)

  return X_train, y_train


def imba_wss(partial_set, imba_type, optimal_alpha, optimal_tol, random_seed):

  # Load partial dataset
  X_train, y_train, X_test, y_test = load_dataset('partial', partial_set, 'tf_idf')

  # Determine whether to use 'weighting' and initiate a linear SVM mdel
  if imba_type == 'weighting':
    weight = int((len(y_train) - y_train.sum()) / y_train.sum())
    clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", tol=optimal_tol, alpha=optimal_alpha,
                                     shuffle=True, max_iter=10000, random_state=1, class_weight={1: weight})
  else:
    clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", tol=optimal_tol, alpha=optimal_alpha,
                                     shuffle=True, max_iter=10000, random_state=1)

  # Apply class imbalance technique to the training data
  X_train, y_train = imba_tech(X_train, y_train, imba_type, random_seed)

  # Train the model
  clf.fit(X_train, y_train)

  # Predict the test data
  clf.predict(X_test)

  # Get a list of distance of all documents to the hyperplane (can be negative)
  d_to_hyper = clf.decision_function(X_test)

  # Calculate WSS@95 value
  wss_95 = calculate_wss_95(y_test, d_to_hyper)

  return wss_95


def run_class_imbalance(partial_sets, optimal_alpha, optimal_tol, repeat):
  """

  :param partial_sets: a list of numbers ranging from 1 to 5, indicating which dataset to be used
  :param optimal_alpha: float, the optimal value of alpha
  :param optimal_tol: float, the optimal value of tolerance
  :param repeat: int, number of repetitions for the randomization
  :return: Results (list of WSS@95 values) of experiments to investigate class imbalance technique
  """

  imba_list = []

  # Investigate the following types in the list
  for imba_type in ['none', 'random_under', 'random_over', 'weighting']:

    result_list = []

    # Split the implementation of different imbalance types
    if imba_type == 'random_under' or imba_type == 'random_over':

      # This loop gets the results (list of WSS@95) of different datasets. For each dataset, repeat certain times.
      for partial_set in partial_sets:

        for seed in range(repeat):

          wss_95 = imba_wss(partial_set, imba_type, optimal_alpha, optimal_tol, seed)

          result_list.append(wss_95)

    else:
      # This loop gets the results (list of WSS@95) of different datasets.
      for partial_set in partial_sets:

        wss_95 = imba_wss(partial_set, imba_type, optimal_alpha, optimal_tol, 1)

        result_list.append(wss_95)

    # Put four lists together in one big list used for the plotting
    imba_list.append(result_list)

  # Store results of this experiment
  with open(config.CLASS_IMBALANCE_RESULTS, 'wb') as handle:
    pickle.dump(imba_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return imba_list


def plot_class_imbalance():

  # Load results of the experiment for class imbalance
  with (open(config.CLASS_IMBALANCE_RESULTS, "rb")) as openfile:
    imba_list = pickle.load(openfile)

  # Plot boxplot
  plt.boxplot(imba_list)

  plt.xticks((1, 2, 3, 4), ('none \n n = 5', 'random \n undersampling \n n = 50',
                            'random \n oversampling \n n = 50', 'weighting \n n = 5'))
  plt.ylabel('WSS@95')

  plt.ylim([-0.05, 0.5])

  # A list of colors used for labelling different datasets
  color = ['r', 'g', 'b', 'y', 'c']

  for i in range(len(imba_list)):

    y = imba_list[i]

    if i == 0 or i == 3:

      x = list(np.random.normal(i + 1, 0.04, size=len(y)))

      for j in range(5):

        plt.scatter(x[j], y[j], c=color[j], alpha=0.6, s=16)

    else:

      x = list(np.random.normal(i + 1, 0.04, size=len(y)))

      plt.scatter(x[:10], y[:10], c=color[0], alpha=0.6, s=16)

      for j in range(5):
        print(j)
        plt.scatter(x[10 * j: 10 * j + 10], y[10 * j: 10 * j + 10], c=color[j], alpha=0.6, s=16)

  # plt.show()
  plt.savefig(config.CLASS_IMBALANCE_FIGURE)
  plt.close()