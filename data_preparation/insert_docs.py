import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import config
import Libs.qrel_parser as qrel_parser
import Database.db_inserts as db_inserts
from data_preparation import qrel_common


def run_qrel_file(filename, start_batch_num=1, running_total=0, max_articles=None):
  connector = db_inserts.Connector()
  parser = qrel_parser.QrelParser(filename)
  batch_num = start_batch_num

  while not parser.isEnd():
    if qrel_common.over_limit(running_total, max_articles):
      break
    batch = parser.getBatch(size=qrel_common.next_batch_size(running_total, max_articles))

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
    if qrel_common.over_limit(running_total, max_articles):
      break

  return batch_num, running_total


if __name__ == '__main__':
  limit = qrel_common.effective_fetch_limit()
  total = 0
  batch_num, total = run_qrel_file(
      config.TRAIN_QREL_LOCATION, start_batch_num=1, running_total=total, max_articles=limit)
  if limit is None or total < limit:
    run_qrel_file(
        config.TEST_QREL_LOCATION, start_batch_num=batch_num, running_total=total, max_articles=limit)
