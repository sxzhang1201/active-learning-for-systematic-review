import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import Libs.file_storage as fs


def boxplot():
  pass


def get_OMB_list(result):

  # load Yield and Burden list
  burden_al = result['burden']
  yield_al = result['yield']

  # Reason for if-else:
  # As we did not record Yield and Burden for the last iteration,
  # but review 'CD009020' and  'CD009185' retrieved one relevant doc at their last active learning iteration.
  if max(yield_al) < 1:
    burden = burden_al[-1]
    print("burden:", burden)

  else:
    # Index of the time point when the Yield first time reach required recall
    index = np.where(np.array(yield_al) == 1)[0]
    burden = burden_al[index[0]]

  # Calculate One Minus Burden (OMB) at that time point
  performance = 1 - burden

  return performance


if __name__ == '__main__':

  baseline_performance = []

  # I only list the reviews of those having results
  analyse_reviews = ['CD007394', 'CD007427', 'CD008686', 'CD008691', 'CD009020',
                     'CD009185', 'CD009372', 'CD009519', 'CD009551']

  for review in analyse_reviews:

    print(review)

    # A list of ten OMB values for ten repetitions
    performance_by_review = []

    results = fs.open_results(review_name = review, run_similarity = False)['results']

    for result in results:
      performance_by_review. append(get_OMB_list(result))

    baseline_performance.append(performance_by_review)

    # Append average OMB of ten repetitions
    # OMB_Baseline.append(sum(OMB_list_by_review)/len(OMB_list_by_review))

  plt.boxplot(baseline_performance)
  # plt.xticks((range(1, len(analyse_reviews) + 1)), (analyse_reviews))

  plt.ylabel('(1-Burden) at 100% Yield')

  plt.show()



