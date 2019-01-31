import Database.db_queries as db

import Libs.preprocess as preprocess
import Libs.feature_extraction as feature_extraction
import Libs.file_storage as fs

documents = db.get_articles()
document_collection = preprocess.clean(documents)

document_collection = fs.load_documents()
matrix_collection = fs.load_matrix_collection()

_, matrix_collection = feature_extraction.vectorize(document_collection)

feature_extraction.get_cosine_similarity(matrix_collection)