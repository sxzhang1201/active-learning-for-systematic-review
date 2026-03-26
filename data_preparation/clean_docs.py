"""
Build preprocessed document vectors from the SQLite DB:
  - Text_data/document_collection.pickle
  - Text_data/matrix_collection.pickle
  - Database/term_document_matrix.pickle (sparse tf matrix for dataset_split)
  - Text_data/cosine_sim_matrix.pickle (for similarity experiments)
"""
import os
import sys
import pickle

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import Database.db  # noqa: F401
import Database.db_queries as db
import Libs.preprocess as preprocess
import Libs.feature_extraction as feature_extraction
import config


def run_clean_and_vectorize():
  config.ensure_output_dirs()
  documents = list(db.get_articles())
  if not documents:
    raise RuntimeError(
        "No articles in the database. Run fetch_raw.py, then insert_docs.py, then re-run this script."
    )

  document_collection = preprocess.clean(documents)
  tf_matrix, matrix_collection = feature_extraction.vectorize(document_collection)

  os.makedirs(os.path.dirname(config.DOC_TERM_MATRIX), exist_ok=True)
  with open(config.DOC_TERM_MATRIX, 'wb') as handle:
    pickle.dump(tf_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

  feature_extraction.get_cosine_similarity(matrix_collection)


if __name__ == '__main__':
  run_clean_and_vectorize()
