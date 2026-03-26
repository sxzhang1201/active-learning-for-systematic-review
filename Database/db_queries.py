from peewee import *
import Database.db as db
import config


def get_articles():
  q = db.Article.select(db.Article.title, db.Article.abstract).order_by(db.Article.id)
  if config.DEBUG is True:
    q = q.limit(10)
  return q.execute()


def get_pmids():
  if config.DEBUG is True:
    pmids = db.Article.select(db.Article.pubmed_id).limit(10).execute()
  else:
    pmids = db.Article.select(db.Article.pubmed_id).execute()

  return [x.pubmed_id for x in pmids]


def get_labels():
  q = db.Article.select(db.Article.included).order_by(db.Article.id)
  if config.DEBUG is True:
    q = q.limit(10)
  return [x.included for x in q.execute()]


def get_review_names():
  q = db.Article.select(db.Article.review_id).order_by(db.Article.id)
  if config.DEBUG is True:
    q = q.limit(10)
  return [x.review_id for x in q.execute()]


def get_review_indices():
  """Contiguous (start, end) row slices per review_id, in Article id order."""
  q = db.Article.select(db.Article.review_id).order_by(db.Article.id)
  if config.DEBUG is True:
    q = q.limit(10)
  review_ids = [x.review_id for x in q.execute()]
  if not review_ids:
    return []
  indices = []
  first = 0
  for i in range(len(review_ids) - 1):
    if review_ids[i] != review_ids[i + 1]:
      indices.append((first, i + 1))
      first = i + 1
  indices.append((first, len(review_ids)))
  return indices