import config
import pickle
import Libs.file_storage as file_handle

class DocumentCollection(object):
  def __init__(self):
    self.documents = []
    self.review_names = []


  def get_labels(self):
    labels = [x.label for x in self.documents]

    return labels


  def get_documents(self):
    return self.documents


  def get_num_docs(self):
    return len(self.documents)


  def get_num_relevant_docs(self):
    relevant_docs = [x for x in self.documents if x.label is 1]

    return len(relevant_docs)


  def get_pubmed_ids(self):
    return [x.pmid for x in self.documents]


  def get_review_names(self):
    return self.review_names


  def get_content(self):
    return [x.content for x in self.documents]


  def get_by_review(self, review_name):
    return [x for x in self.documents if x.review_name is review_name]


  def add_document(self, new_document):
    if new_document.review_name not in self.review_names:
      self.review_names.append(new_document.review_name)

    self.documents.append(new_document)


class Document(object):
  def __init__(self, content, pmid, review_name, label = 0):
    self.content = content
    self.pmid = pmid
    self.label = label
    self.review_name = review_name