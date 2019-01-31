from multiprocessing import Pool
from functools import partial
import Libs.file_storage as fs
import numpy as np
import config
import Libs.classifier as classifier

print('loading data objects')
documents = fs.load_documents()
matrix_collection = fs.load_matrix_collection()

label_list = documents.get_labels()

# Determined by their characteristics
removed_reviews = ['CD011549', 'CD010386', 'CD012019', 'CD010633']

# Full set of reviews
review_names = matrix_collection.get_review_names()
# Remove the reviews that were excluded based on their characteristics
review_names = list(set(review_names) - set(removed_reviews))

alpha_range = list(10.0 ** -np.arange(-1, 7))
tolerance_range = list(10.0 ** -np.arange(-1, 7))

grid_parameters = {'alpha': alpha_range, 'tol': tolerance_range}

print('setting up ModelBuilder')
clf = classifier.ModelBuilder(matrix_collection, review_names, label_list, grid_parameters)

print('starting parallel processing')
# Fix parameters of run_fold function
parallel_func = partial(clf.run_review)

with Pool(processes = config.NUM_CORES) as pool:
  # Parallelise the iterations
  pool.map(parallel_func, review_names)