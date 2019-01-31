import pandas

class QrelParser(object):

  pointer = 0
  qrel_header = []

  # Sets the qrel header names and qrel filename
  def __init__(self, filename = None, qrels_header = ['review_id', 'q0', 'pmid', 'included']):
    self.qrels_header = qrels_header

    if filename:
      self.readFile(filename, qrels_header)


  # Read qrel file into pandas dataframe
  def readFile(self, filename, qrels_header):
    self.filename = filename
    self.qrel_data = pandas.read_csv(self.filename, sep = "\s+", names = qrels_header)


  # Fetch a batch of size N from the read qrel file
  def getBatch(self, size = 50000, offset = None):
    if offset is not None:
      self.pointer = offset
    
    batch = self.qrel_data[self.pointer:self.pointer + size]
    self.pointer = self.pointer + size

    return batch


  # Get PubMed IDs for current batch
  def getBatchPMIDs(self, batch):
    pmids = batch['pmid']

    return [str(s) for s in pmids]


  # Check whether the pointer is at the end of the 
  # current qrel file data
  def isEnd(self):
    return self.pointer >= len(self.qrel_data)


  # Get all PubMed IDs for the current qrel file
  def getAllPMIDs(self):
    pmid_list = list(self.qrel_data['pmid'])

    return [str(s) for s in pmid_list]