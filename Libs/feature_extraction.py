from sklearn.feature_extraction.text import CountVectorizer
import Libs.Preprocessing.create_objects as collection_creator
from sklearn.metrics.pairwise import cosine_similarity
import nltk, csv, numpy as np, pandas as pd
import Libs.file_storage as fs


def vectorize(document_collection):
  """
  Convert a collection of text documents to a matrix of token counts (term frequency).

  :param corpus: List of strings, e.g.['the evaluation of ct test', 'the analysis of mri', 'deal with machine learning']
  :param min_freq: A threshold that terms whose document frequency is lower than will be ignored
  :param max_freq: A threshold that terms whose document frequency is higher than will be ignored

  :return: Feature vector space and feature's index loop-up dictionary
  """

  corpus = document_collection.get_content()

  min_freq = document_collection.get_num_relevant_docs() * 0.01
  max_freq = document_collection.get_num_docs() * 0.9

  # Initiate token-count based vectorizer
  vectorizer = CountVectorizer(min_df = min_freq, max_df = max_freq)

  # Learn all tokens in the corpus
  vectorizer.fit(corpus)

  # Tranform documents to a document-term matrix
  tf_matrix = vectorizer.transform(corpus)

  # Create and store a matrix collection class
  matrix_collection = collection_creator.create_matrix_collection(tf_matrix, document_collection)

  return tf_matrix, matrix_collection


def word_freq(split_corpus, path):
  """
  Obtain the word frequency table and export as a csv file
  :param split_corpus: a list that contain all the words, e.g. ['abu', 'bili', 'pregnan', ...]
  :param path: string, the path for the csv file
  :return: term-frequency matrix, e.g.  [('abu', 1000), ('bili', 250), ('pregnan', 150), ...]
  """

  # Get Word Frequency
  term_freq = dict(nltk.FreqDist(split_corpus))

  # Export Word Frequency as a csv file
  with open(path, 'w') as handle:
    writer = csv.writer(handle)
    writer.writerows(term_freq.items())

  return term_freq


def get_cosine_similarity(matrix_collection):
  matrix = matrix_collection.get_feature_matrix()
  review_indices = matrix_collection.get_review_indices()
  labels = matrix_collection.get_review_names()
  
  review_means = list()

  # Loop over the review indices
  for l,r in review_indices:
    # Get the subset of review documents
    review_subset = matrix[l:r,:]

    # Get the mean vector for this review
    review_means.append(np.mean(review_subset, axis = 0))

  # Stack the mean vectors of all reviews into a matrix
  review_matrix = np.vstack(review_means)

  # Calculate the cosine similarity between all reviews
  cos_sim = cosine_similarity(review_matrix)
  
  # Put into a pandas dataframe where we can add row and
  # column labels with the review label (e.g. CD010276)
  df = pd.DataFrame(cos_sim, index = labels, columns = labels)

  # Store as a pickle file and csv file
  fs.store_matrix('cosine_sim_matrix', df)
  # df.to_csv(config.TEXT_DATA_LOCATION + '/cosine_sim_matrix.csv')

  return cos_sim