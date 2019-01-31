from PmedConnect import PubmedAPI as api
import config, pickle, Libs.qrel_parser as qrel_parser, Database.db_inserts as db

def run_qrel_file(filename, start_batch_num = 1):
  connector = db.Connector()
  parser = qrel_parser.QrelParser(filename)
  
  batch_num = start_batch_num

  while not parser.isEnd():
    batch = parser.getBatch()

    with (open(config.STORE_LOCATION + '_' + str(batch_num), 'rb')) as article_file:
      articles = pickle.load(article_file)

      connector.insert_fetched_articles(articles, batch)

    batch_num += 1

  return batch_num

if __name__ == '__main__':
  # Run both the train and test qrel files to fetch
  # all (i.e. currently 50) systematic review results
  batch_num = run_qrel_file(config.TRAIN_QREL_LOCATION)
  run_qrel_file(config.TEST_QREL_LOCATION, batch_num)