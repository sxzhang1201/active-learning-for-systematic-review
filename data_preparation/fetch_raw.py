from PmedConnect import PubmedAPI as api
import config, pickle, Libs.qrel_parser as qrel_parser

def run_qrel_file(filename, start_batch_num = 1):
  searcher = api.PubmedAPI(config.PUBMED_EMAIL)

  parser = qrel_parser.QrelParser(filename)
  
  batch_num = start_batch_num

  while not parser.isEnd():
    batch = parser.getBatch()
    
    pubmed_ids = parser.getBatchPMIDs(batch)

    articles = searcher.fetch(pubmed_ids)

    with (open(config.STORE_LOCATION + '_' + str(batch_num), 'wb')) as store_file:
      pickle.dump(articles, store_file)
    
    batch_num += 1

  return batch_num

if __name__ == '__main__':
  # Run both the train and test qrel files to fetch
  # all (i.e. currently 50) systematic review results
  batch_num = run_qrel_file(config.TRAIN_QREL_LOCATION)
  run_qrel_file(config.TEST_QREL_LOCATION, batch_num)