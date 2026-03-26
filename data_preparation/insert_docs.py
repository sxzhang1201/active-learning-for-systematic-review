import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import pickle
import config
import Libs.qrel_parser as qrel_parser
import Database.db_inserts as db

_DEFAULT_BATCH = 50000


def run_qrel_file(filename, start_batch_num=1, running_total=0, max_articles=None):
  """
  Must follow the same qrel batching as fetch_raw.py so batch numbers align with
  Pubmed results/_N pickle files.
  """
  connector = db.Connector()
  parser = qrel_parser.QrelParser(filename)

  batch_num = start_batch_num

  while not parser.isEnd():
    if max_articles is not None and running_total >= max_articles:
      break

    batch_size = _DEFAULT_BATCH
    if max_articles is not None:
      batch_size = min(batch_size, max(1, max_articles - running_total))

    batch = parser.getBatch(size=batch_size)

    pickle_path = config.STORE_LOCATION + '_' + str(batch_num)
    if not os.path.isfile(pickle_path):
      raise FileNotFoundError(
          'Missing PubMed batch file:\n  {}\n'
          'Run data_preparation/fetch_raw.py first with the *same* FETCH_RAW_MAX_ARTICLES '
          'as in config.py (this insert step must mirror fetch batching).'.format(pickle_path))

    with open(pickle_path, 'rb') as article_file:
      articles = pickle.load(article_file)

    connector.insert_fetched_articles(articles, batch)

    running_total += len(articles)
    batch_num += 1

    if max_articles is not None and running_total >= max_articles:
      break

  return batch_num, running_total


if __name__ == '__main__':
  limit = config.FETCH_RAW_MAX_ARTICLES or None
  if limit is not None and limit <= 0:
    limit = None
  total = 0
  batch_num, total = run_qrel_file(
      config.TRAIN_QREL_LOCATION, start_batch_num=1, running_total=total, max_articles=limit)
  if limit is None or total < limit:
    run_qrel_file(
        config.TEST_QREL_LOCATION, start_batch_num=batch_num, running_total=total, max_articles=limit)
