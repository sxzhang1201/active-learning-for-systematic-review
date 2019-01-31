import config
import re


def filter_html(article):
  # PubMed articles may contain the following HTML tags: <i>, <u>, <b>, <sup>, and <sub>
  # Remove all the possible HTML tags from the text, replace them with a single space
  for tag_name in ['i', 'u', 'b', 'sup', 'sub']:
    if '<%s>' % tag_name in article:
      article = article.replace('<%s>' % tag_name, ' ')
      
    if '</%s>' % tag_name in article:
      article = article.replace('</%s>' % tag_name, ' ')

  return article


def replace_dashes(article):
  replace = r'\1 \2'

  if config.KEEP_DASHES is True:
    replace = r'\1_\2'

  # Replace dashes that connect two characters with an underscore (- to _)
  article = re.sub(r'([a-z])\-([a-z])', replace, article, 0, re.IGNORECASE)

  return article


def remove_special_characters(article):
  # Remove any character that is not alphanumeric or an underscore (_)
  article = re.sub(r'[^a-z0-9_]', ' ', article, 0, re.IGNORECASE)

  return article


def remove_double_spaces(article):
  # Replace any multiple occurences of spaces with a single space
  article = re.sub(r'\s{2,}', ' ', article).strip()

  return article


def replace_numbers(article):
  replace = ' '

  if config.KEEP_NUMBERS is True:
    replace = '_number_'

  # Replace any number with a placeholder (_number_)
  article = re.sub(r'[0-9]+', replace, article)

  return article


def clean_item(item):
  if item is None or len(item) < 1:
    return ''

  item = filter_html(item)
  item = replace_dashes(item)
  item = remove_special_characters(item)
  item = remove_double_spaces(item)
  item = replace_numbers(item)

  # Convert to lowercase
  item = item.lower()

  return item


def remove_all(documents):
  documents = list(map(clean_item, documents))

  return documents