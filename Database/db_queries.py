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
  review_id_query = db.Article.select(db.Article.review_id).order_by(db.Article.id).execute()

  review_ids = [x.review_id for x in review_id_query]

  indices = []

  last_review = None
  first_spotted = 0

  for i in range(len(review_ids)):
    review_id = review_ids[i]

    if last_review is None:
      last_review = review_id

    if i < len(review_ids) - 1:
      next_review_id = review_ids[i]
      
      if next_review_id != last_review:
        indices.append((first_spotted, i))

        first_spotted = i
        last_review = next_review_id
    else:
      indices.append((first_spotted, i + 1))

  return indices