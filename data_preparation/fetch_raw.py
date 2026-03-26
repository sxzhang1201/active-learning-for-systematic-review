import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import config
import pickle
import Libs.qrel_parser as qrel_parser
from Libs.pubmed_fetch import PubmedAPI

_DEFAULT_BATCH = 50000


def run_qrel_file(filename, start_batch_num=1, running_total=0, max_articles=None):
  """
  :param max_articles: if set, stop after this many qrel rows total (including running_total);
         uses small getBatch slices so the qrel pointer stays consistent.
  """
  searcher = PubmedAPI(config.PUBMED_EMAIL)

  parser = qrel_parser.QrelParser(filename)

  batch_num = start_batch_num

  while not parser.isEnd():
    if max_articles is not None and running_total >= max_articles:
      break

    batch_size = _DEFAULT_BATCH
    if max_articles is not None:
      batch_size = min(batch_size, max(1, max_articles - running_total))

    batch = parser.getBatch(size=batch_size)

    pubmed_ids = parser.getBatchPMIDs(batch)

    articles = searcher.fetch(pubmed_ids)

    with (open(config.STORE_LOCATION + '_' + str(batch_num), 'wb')) as store_file:
      pickle.dump(articles, store_file)

    running_total += len(pubmed_ids)
    batch_num += 1

    if max_articles is not None and running_total >= max_articles:
      break

  return batch_num, running_total


if __name__ == '__main__':
  limit = config.FETCH_RAW_MAX_ARTICLES or None
  if limit is not None and limit <= 0:
    limit = None
  if limit is not None:
    print('FETCH_RAW_MAX_ARTICLES={} (set to 0 in config or env for full run)'.format(limit))
  total = 0
  batch_num, total = run_qrel_file(
      config.TRAIN_QREL_LOCATION, start_batch_num=1, running_total=total, max_articles=limit)
  if limit is None or total < limit:
    run_qrel_file(
        config.TEST_QREL_LOCATION, start_batch_num=batch_num, running_total=total, max_articles=limit)