from peewee import *
import config, datetime

conn = SqliteDatabase(config.DB_FILE)

class BaseModel(Model):
  class Meta:
    database = conn


class Journal(BaseModel):
  title = CharField(max_length = 150)
  iso = CharField(max_length = 100)
  iso_stripped = CharField(max_length = 100)
  issn = CharField(max_length = 45, null = True)
  created_date = DateTimeField(default = datetime.datetime.now)


class Article(BaseModel):
  pubmed_id = CharField(max_length = 45)
  title = CharField(max_length = 300)
  title_stripped = CharField(max_length = 300)
  abstract = TextField(null = True)
  journal = ForeignKeyField(Journal, null = True)
  publication_date = DateField(null = True)
  doi = CharField(max_length = 45, null = True)
  created_date = DateTimeField(default = datetime.datetime.now)
  included = BooleanField()
  review_id = CharField(max_length = 10)


class Keyword(BaseModel):
  keyword = CharField(max_length = 45)
  created_date = DateTimeField(default = datetime.datetime.now)


class Keyword_to_article(BaseModel):
  article = ForeignKeyField(Article)
  keyword = ForeignKeyField(Keyword)
  created_date = DateTimeField(default = datetime.datetime.now)

  class Meta:
    # Create compound primary key with article and keyword
    index = (
      (('article', 'keyword'), True),
    )


class Search_results(BaseModel):
  pubmed_id = CharField(max_length = 45)
  search_query = CharField(max_length = 200)
  fetched = BooleanField(default = False)
  created_date = DateTimeField(default = datetime.datetime.now)


# Instantiate the database
if not Article.table_exists():
  conn.create_tables([Journal, Article, Keyword, Keyword_to_article, Search_results])

def max_sql_variables():
  # Get the maximum number of arguments allowed in
  # a query by the current sqlite3 implementation
  import sqlite3
  db = sqlite3.connect(':memory:')
  
  cur = db.cursor()
  cur.execute('CREATE TABLE t (test)')
  
  low, high = 0, 100000

  while (high - 1) > low: 
    guess = (high + low) // 2
    query = 'INSERT INTO t VALUES ' + ','.join(['(?)' for _ in range(guess)])
    args = [str(i) for i in range(guess)]

    try:
      cur.execute(query, args)
    except sqlite3.OperationalError as e:
      if 'too many SQL variables' in str(e):
        high = guess
      else:
        raise
    else:
      low = guess

  cur.close()
  db.close()
  
  return low