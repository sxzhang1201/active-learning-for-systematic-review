def remove_short_tokens(item):
  # Delete any tokens longer than 34 characters
  item = [w for w in item if len(w) < 35]

  return item


def remove_long_tokens(item):
  # Delete any tokens smaller than 2 characters
  item = [w for w in item if len(w) > 1]

  return item


def remove_tokens_by_size(item):
  item = remove_short_tokens(item)
  item = remove_long_tokens(item)

  return item


def remove_all(documents):
  documents = list(map(remove_tokens_by_size, documents))

  return documents