from nltk.tokenize import WhitespaceTokenizer


def tokenize(documents):
  tokenizer = WhitespaceTokenizer()

  def tokenize_doc(document):
    return tokenizer.tokenize(document)

  """
  Ingests content, converts to lowercase, removes special characters except for _, ?, and %,
  replaces dashes and hypens. Returns full or unique list of cleaned words in content.

  :param content: String of text to tokenize.
  :param unique: Boolean indicating whether to make the output list of words unique or not.
  :return: list of cleaned and tokenized input content.
  """

  if documents is None:
    return None

  documents = list(map(tokenize_doc, documents))

  # Return an occurrence matrix instead of a frequency matrix
  # if unique is True:
  #   # set() removes duplicates and returns a dict, convert back into list
  #   words = list(set(words))
  
  return documents


def convert_to_plain_text(documents):
  plain_text = []

  for document in documents:
    if document.abstract is None:
      document.abstract = ''

    doc_string = (document.title + ' ' + document.abstract)

    # Convert content into lowercase string
    doc_string.lower()

    plain_text.append(doc_string)

  return plain_text