import Database.document as doc_class
import Database.feature_matrix as matrix_class
import Database.db_queries as db
import Libs.file_storage as file_handle


def create_document_collection(documents):
  labels = db.get_labels()
  pmids = db.get_pmids()
  review_names = db.get_review_names()

  doc_collection = doc_class.DocumentCollection()

  for i in range(0, len(documents)):
    doc = documents[i]
    label = labels[i]
    pmid = pmids[i]
    review_name = review_names[i]

    doc_item = doc_class.Document(doc, pmid, review_name, label)

    doc_collection.add_document(doc_item)

  file_handle.store_documents(doc_collection)

  return doc_collection


def create_matrix_collection(matrix, document_collection):
  matrix_collection = matrix_class.Feature_matrix()

  review_indices = db.get_review_indices()
  print(review_indices)
  pubmed_ids = document_collection.get_pubmed_ids()
  review_names = document_collection.get_review_names()

  matrix_collection.insert_matrix(matrix, review_indices, review_names, pubmed_ids)

  file_handle.store_matrix_collection(matrix_collection)

  return matrix_collection