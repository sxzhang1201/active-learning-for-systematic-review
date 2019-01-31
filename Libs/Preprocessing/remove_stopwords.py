from nltk.corpus import stopwords
import config, nltk

# Check whether the stopwords data is already downloaded
# If not, download the stopword data
try:
  nltk.data.find('corpora/stopwords')
except LookupError:
  nltk.download('stopwords')

# Merge nltk defined stopwords with manually defined stopwords
stopwords = set(stopwords.words('english') + config.EXTRA_STOPWORDS)


def remove_stopwords(item):
  item = [w for w in item if w not in stopwords]

  return item


def remove_all(documents):
  documents = list(map(remove_stopwords, documents))

  return documents