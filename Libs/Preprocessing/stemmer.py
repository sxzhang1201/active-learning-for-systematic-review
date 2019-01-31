import nltk.stem as stemming
from nltk.stem import WordNetLemmatizer

class WordNetStemmer(WordNetLemmatizer):
  """Implementation of the lemmatize stemmer."""

  def stem(self, word, pos = u'n'):
    return self.lemmatize(word, pos)


class Stemmer(object):
  """
  Define Class Stemmer that can be instantiated with three different types: Porter, Snowball, and Lemmatize.
  """

  def __init__(self, stemmer_type):
    if stemmer_type == 'porter':
      self.stemmer = stemming.PorterStemmer()
    elif stemmer_type == 'snowball':
      self.stemmer = stemming.SnowballStemmer('english')
    elif stemmer_type == 'lemmatize':
      self.stemmer = WordNetStemmer()
    else:
      raise NameError('\'%s\' not supported' % stemmer_type)


def stem(documents, stemmer_type = 'porter'):
  def stem_document(document):
    # Loop over the vocabulary and stem each word, return as list
    return [stemmer.stem(d) for d in document]

  # Instantiate the stemmer class
  stemmer = Stemmer(stemmer_type).stemmer

  stemmed_documents = list(map(stem_document, documents))

  return stemmed_documents