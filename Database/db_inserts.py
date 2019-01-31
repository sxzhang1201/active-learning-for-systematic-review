import Database.db as db
from set_up import config
import progressbar, re
from peewee import *

db_obj = SqliteDatabase(config.DB_FILE)

class Connector:

  inserts = 0
  SQLITE_MAX_VARIABLE_NUMBER = None


  def insert_search_results(self, pubmed_ids, query):
    data_list = []

    for pubmed_id in pubmed_ids:
      data_list.append(dict(pubmed_id = pubmed_id, search_query = query))

    with db_obj.atomic():
      for idx in range(0, len(data_list), config.DB_INSERT_LIMIT):
        db.Search_results.insert_many(data_list[idx:idx + config.DB_INSERT_LIMIT]).execute()


  def set_search_results_fetched(self, pubmed_ids):
    query = db.Search_results.update(fetched = True).where(db.Search_results.pubmed_id << pubmed_ids)
    query.execute()


  def get_unfetched_search_results(self):
    result = db.Search_results.select(db.Search_results.pubmed_id).where(db.Search_results.fetched == False)

    return [i.pubmed_id for i in result]

  def insert_fetched_articles(self, articles, batch):
    if self.SQLITE_MAX_VARIABLE_NUMBER is None:
      self.SQLITE_MAX_VARIABLE_NUMBER = db.max_sql_variables()

    pubmed_ids = []
    article_list = []

    print('Inserting %i articles.' % len(articles))

    bar = progressbar.ProgressBar()

    data_list = []

    for idx in bar(range(0, len(articles))):
      article_data = articles[idx]
      batch_item = batch.iloc[idx]
      
      journal_id = None

      if article_data['journal_title'] is not None:
        journal_id = self.get_create_journal(article_data)

      publication_date = self.get_publication_date(article_data)

      try:
        data_list.append({
          'pubmed_id': article_data['pmid'],
          'title': article_data['title'],
          'title_stripped': re.sub('[^a-z]', '', article_data['title'].lower()),
          'abstract': article_data['abstract'],
          'journal': journal_id,
          'publication_date': publication_date,
          'doi': article_data['doi'],
          'included': batch_item['included'],
          'review_id': batch_item['review_id'],
        })
      except (AttributeError):
        print(idx)
        print(article_data)
        quit()

    print('\n')

    with db.conn.atomic():
      # remove one to avoid issue if peewee adds some variable
      insert_size = (self.SQLITE_MAX_VARIABLE_NUMBER // (len(data_list[0]) + 1))

      for idx in range(0, len(data_list), insert_size):        
        db.Article.insert_many(data_list[idx:idx + insert_size]).execute()


  def insert_keywords(self, article_id, keywords):
    for keyword in keywords:
      keyword, created = db.Keyword.get_or_create(keyword = keyword)

      db.Keyword_to_article.create(article_id = article_id, keyword_id = keyword.id)


  def get_create_journal(self, article):
    journal = {
      'title': article['journal_title'],
      'iso': article['journal_iso'],
      'iso_stripped': re.sub('[^a-z]', '', article['journal_iso']),
      'issn': article['journal_issn'],
    }

    try:
      found_journal = db.Journal.get(
          (db.Journal.title == journal['title'])
          or (db.Journal.iso == journal['iso'])
          or (db.Journal.iso_stripped == journal['iso_stripped'])
          or (db.Journal.issn == journal['issn']))
    except db.Journal.DoesNotExist:
      found_journal = db.Journal.create(**journal)

    return found_journal.id


  def get_publication_date(self, article):
    if article['pub_year'] is not None and article['pub_month'] is not None and article['pub_day'] is not None:
      return "%s-%s-%s" % (article['pub_year'], article['pub_month'].zfill(2), article['pub_day'].zfill(2))

    if article['date_pubmed_published'] is not None:
      return article['date_pubmed_published']

    if article['date_medline_published'] is not None:
      return article['date_medline_published']