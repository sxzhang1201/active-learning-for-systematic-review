import pandas
import Database.db as db, config, qrel_parser

if __name__ == "__main__":
  train_parser = qrel_parser.QrelParser(config.TRAIN_QREL_LOCATION)
  test_parser = qrel_parser.QrelParser(config.TEST_QREL_LOCATION)

  train_clef_data = train_parser.qrel_data
  test_clef_data = test_parser.qrel_data

  frames = [train_clef_data, test_clef_data]
  clef_data = pandas.concat(frames)

  pubmed_data = db.Article.select()

  for idx in range(0, len(clef_data)):
    clef_item = clef_data.iloc[idx]
    pubmed_item = pubmed_data[idx]

    if str(clef_item['review_id']) != str(pubmed_item.review_id):
      print('review_id error, id: %i, clef: %s, db: %s.' % (idx, clef_item['review_id'], pubmed_item.review_id))
      quit()

    if str(clef_item['pmid']) != str(pubmed_item.pubmed_id):
      print('pubmed_id error, id: %i, clef: %i, db: %s.' % (idx, clef_item['pmid'], pubmed_item.pubmed_id))
      quit()

    if bool(clef_item['included']) != bool(pubmed_item.included):
      print('included error, id: %i' % (idx))
      print(clef_item)
      print(pubmed_item.id)
      quit()