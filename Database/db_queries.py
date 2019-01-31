from peewee import *
import Database.db as db
import config


def get_articles():
  if config.DEBUG is True:
    return db.Article.select(db.Article.title, db.Article.abstract).limit(10).execute()
  else:
    return db.Article.select(db.Article.title, db.Article.abstract).execute()


def get_pmids():
  if config.DEBUG is True:
    pmids = db.Article.select(db.Article.pubmed_id).limit(10).execute()
  else:
    pmids = db.Article.select(db.Article.pubmed_id).execute()

  return [x.pubmed_id for x in pmids]


def get_labels():
  if config.DEBUG is True:
    labels = db.Article.select(db.Article.included).limit(10).execute()
  else:
    labels = db.Article.select(db.Article.included).execute()

  return [x.included for x in labels]


def get_review_names():
  if config.DEBUG is True:
    review_id_query = db.Article.select(db.Article.review_id).limit(10).execute()
  else:
    review_id_query = db.Article.select(db.Article.review_id).execute()  

  review_ids = [x.review_id for x in review_id_query]

  return review_ids


def get_review_indices():
  review_id_query = db.Article.select(db.Article.review_id).execute()

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