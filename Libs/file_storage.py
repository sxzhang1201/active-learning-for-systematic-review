import config, pickle


def store_classifier(i, feature_type, classifier_obj, dataset_num = None):
  dataset_name = 'classifier'

  if feature_type == 'tm':
    dataset_name = 'classifier_%i' % dataset_num

  with open(config.CLASSIFIER_LOCATION + '/%i_%s_%s.pickle' % (i, feature_type, dataset_name), 'wb') as handle:
    pickle.dump(classifier_obj, handle, protocol = pickle.HIGHEST_PROTOCOL)


def load_classifier(i, feature_type, dataset_num = None):
  dataset_name = 'classifier'

  if feature_type == 'tm':
    dataset_name = 'classifier_%i' % dataset_num

  filename = config.CLASSIFIER_LOCATION + '/%i_%s_%s.pickle' % (i, feature_type, dataset_name)

  if file_exists(filename):
    return read_file(filename)

  return None


def store_documents(document_collection):
  with open(config.DOC_COLLECTION, 'wb') as handle:
    pickle.dump(document_collection, handle, protocol = pickle.HIGHEST_PROTOCOL)


def load_documents():
  if file_exists(config.DOC_COLLECTION):
    return read_file(config.DOC_COLLECTION)

  return None


def store_matrix_collection(matrix_collection):
  with open(config.MATRIX_COLLECTION, 'wb') as handle:
    pickle.dump(matrix_collection, handle, protocol = pickle.HIGHEST_PROTOCOL)


def load_matrix_collection():
  if file_exists(config.MATRIX_COLLECTION):
    return read_file(config.MATRIX_COLLECTION)

  return None


def store_results(review_name, data, run_similarity = False, similarity_size = None):
  dataset_name = ('review_%s' % review_name)

  if run_similarity is False:
    dataset_name = dataset_name + '_leaveoneout'
  else:
    dataset_name = dataset_name + ('_similarity_size_%i' % similarity_size)

  dataset_name = config.RESULTS_LOCATION + '/' + dataset_name + '.pickle'
  
  if file_exists(dataset_name):
    old_data = read_file(dataset_name)

    old_data['results'].append(data)

    data = old_data
  else:
    data = {
      'review_name': review_name,
      'results': [data]
    }

  with open(dataset_name, 'wb') as handle:
    pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)


def open_results(review_name, run_similarity = False, similarity_size = None):
  dataset_name = ('review_%s' % review_name)

  if run_similarity is False:
    dataset_name = dataset_name + '_leaveoneout'
  else:
    dataset_name = dataset_name + ('_similarity_size_%i' % similarity_size)

  dataset_name = config.RESULTS_LOCATION + '/' + dataset_name + '.pickle'

  with (open(dataset_name, "rb")) as openfile:
    results = pickle.load(openfile)

  return results

def load_matrix(name):
  filename = config.TEXT_DATA_LOCATION + '/%s.pickle' % name

  if file_exists(filename):
    return read_file(filename)
  
  return None


def store_matrix(name, matrix_object):
  with open(config.TEXT_DATA_LOCATION + '/%s.pickle' % name, 'wb') as handle:
    pickle.dump(matrix_object, handle, protocol = pickle.HIGHEST_PROTOCOL)
    

def file_exists(name):
  try:
    with open(name, 'rb') as handle:
      return handle
  except (FileNotFoundError, OSError, IOError) as e:
    return False


def read_file(name):
  with open(name, 'rb') as handle:
    return pickle.load(handle)