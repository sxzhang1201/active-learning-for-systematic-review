import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import config
import Libs.qrel_parser as qrel_parser
from data_preparation import qrel_common
from Libs.pubmed_fetch import PubmedAPI


def run_qrel_file(filename, start_batch_num=1, running_total=0, max_articles=None):
  searcher = PubmedAPI(config.PUBMED_EMAIL)
  parser = qrel_parser.QrelParser(filename)
  batch_num = start_batch_num

  while not parser.isEnd():
    if qrel_common.over_limit(running_total, max_articles):
      break
    batch = parser.getBatch(size=qrel_common.next_batch_size(running_total, max_articles))
    pubmed_ids = parser.getBatchPMIDs(batch)
    articles = searcher.fetch(pubmed_ids)

    with open(config.STORE_LOCATION + '_' + str(batch_num), 'wb') as store_file:
      pickle.dump(articles, store_file)

    running_total += len(pubmed_ids)
    batch_num += 1
    if qrel_common.over_limit(running_total, max_articles):
      break

  return batch_num, running_total


if __name__ == '__main__':
  limit = qrel_common.effective_fetch_limit()
  if limit is not None:
    print('FETCH_RAW_MAX_ARTICLES={} (set to 0 in config or env for full run)'.format(limit))
  total = 0
  batch_num, total = run_qrel_file(
      config.TRAIN_QREL_LOCATION, start_batch_num=1, running_total=total, max_articles=limit)
  if limit is None or total < limit:
    run_qrel_file(
        config.TEST_QREL_LOCATION, start_batch_num=batch_num, running_total=total, max_articles=limit)
