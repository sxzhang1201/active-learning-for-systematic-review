import os

# Project root is the directory containing this file (not the process cwd).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DB_FILE = os.path.join(PROJECT_ROOT, 'Database', 'miner_database.db')

DEBUG = False

### Pubmed API config:
PUBMED_EMAIL = 's.x.zhang1201@gmail.com'
# Identifies your tool to NCBI (use your own email for PUBMED_EMAIL per NCBI policy).
PUBMED_TOOL_NAME = 'active-learning-for-systematic-review'
# Smaller batches reduce "connection closed without response" errors from NCBI.
PUBMED_CHUNK_SIZE = int(os.environ.get('PUBMED_CHUNK_SIZE', '80'))
PUBMED_EFETCH_RETRIES = int(os.environ.get('PUBMED_EFETCH_RETRIES', '6'))
DB_INSERT_LIMIT = 100

# Limit rows processed by data_preparation/fetch_raw.py (PubMed downloads + pickle batches).
# Use 10 for a quick smoke test; set 0 (or env FETCH_RAW_MAX_ARTICLES=0) for the full corpus.
FETCH_RAW_MAX_ARTICLES = int(os.environ.get('FETCH_RAW_MAX_ARTICLES', '10'))

### CLEF specific config
STORE_LOCATION = os.path.join(PROJECT_ROOT, 'Pubmed results') + os.sep
TRAIN_QREL_LOCATION = os.path.join(PROJECT_ROOT, 'Database', 'CLEF_data', 'qrel_abs_train')
TEST_QREL_LOCATION = os.path.join(PROJECT_ROOT, 'Database', 'CLEF_data', 'qrel_abs_test')
DOC_COLLECTION = os.path.join(PROJECT_ROOT, 'Text_data', 'document_collection.pickle')

TEXT_DATA_LOCATION = os.path.join(PROJECT_ROOT, 'Text_data')

TF_MATRIX = os.path.join(TEXT_DATA_LOCATION, 'tf_matrix.pickle')
MATRIX_COLLECTION = os.path.join(TEXT_DATA_LOCATION, 'matrix_collection.pickle')

RESULTS_LOCATION = os.path.join(PROJECT_ROOT, 'Results')
CLASSIFIER_LOCATION = os.path.join(PROJECT_ROOT, 'Classifiers')

DATASETS_DIR = os.path.join(PROJECT_ROOT, 'Datasets')
CONSOLIDATION_DIR = os.path.join(PROJECT_ROOT, 'Consolidation')
CONSOLIDATION_QUERY_STRATEGY = os.path.join(CONSOLIDATION_DIR, 'query_strategy')
CONSOLIDATION_SAMPLE_WEIGHT = os.path.join(CONSOLIDATION_DIR, 'sample_weight')
CONSOLIDATION_ADDED_VALUE = os.path.join(CONSOLIDATION_DIR, 'added_value')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'Figure')

EXTRA_STOPWORDS = []
KEEP_DASHES = False
KEEP_NUMBERS = False

# Use -1 for all CPUs in joblib.Parallel; 1 disables parallelism.
NUM_CORES = int(os.environ.get('AL_NUM_JOBS', '-1'))
NUMBER_CV = 3
NUM_ITERATIONS = 2

SIMILAR_SIZES = [1, 5, 10]

DOC_TERM_MATRIX = os.path.join(PROJECT_ROOT, 'Database', 'term_document_matrix.pickle')

# Figure:
WORD_FREQUENCY = os.path.join(FIGURE_DIR, 'term_frequency.csv')

# 2. Baseline model design
GRID_SEARCH = os.path.join(CONSOLIDATION_DIR, 'grid_search.pickle')
TF_IDF_RESULTS = os.path.join(CONSOLIDATION_DIR, 'tf_idf_test.pickle')
CLASS_IMBALANCE_RESULTS = os.path.join(CONSOLIDATION_DIR, 'class_imbalance.pickle')

HEAT_MAP = os.path.join(FIGURE_DIR, 'heatmap.pdf')
TF_IDF_FIGURE = os.path.join(FIGURE_DIR, 'tf_idf.pdf')
CLASS_IMBALANCE_FIGURE = os.path.join(FIGURE_DIR, 'class_imbalance.pdf')

# 3. Active learning model design
QUERY_STRATEGY_FIGURE = os.path.join(FIGURE_DIR, 'query_strategy.pdf')
DOCUMENT_WEIGHT = os.path.join(FIGURE_DIR, 'importance.pdf')
RELEVANT_DOCUMENT_WEIGHT = os.path.join(FIGURE_DIR, 'relevant_importance.pdf')

ADDED_VALUE_TREND_FIGURE = os.path.join(FIGURE_DIR, 'Trend_Diff_WSS@100.pdf')


def configure_matplotlib():
  """Non-interactive Agg by default; set MPLBACKEND=TkAgg for GUI."""
  import matplotlib
  backend = os.environ.get('MPLBACKEND', 'Agg')
  matplotlib.use(backend)


def effective_pool_processes():
  """multiprocessing.Pool requires a positive count; NUM_CORES <= 0 uses all logical CPUs."""
  if NUM_CORES <= 0:
    return max(1, os.cpu_count() or 1)
  return NUM_CORES


def ensure_output_dirs():
  """Create directories used for outputs (safe to call repeatedly)."""
  for path in (
      TEXT_DATA_LOCATION,
      RESULTS_LOCATION,
      CLASSIFIER_LOCATION,
      DATASETS_DIR,
      CONSOLIDATION_DIR,
      CONSOLIDATION_QUERY_STRATEGY,
      CONSOLIDATION_SAMPLE_WEIGHT,
      CONSOLIDATION_ADDED_VALUE,
      FIGURE_DIR,
      os.path.dirname(DB_FILE),
  ):
    os.makedirs(path, exist_ok=True)
  # PubMed fetch dir
  os.makedirs(STORE_LOCATION.rstrip(os.sep), exist_ok=True)


def dataset_pickle_path(naming, count, st_type):
  return os.path.join(DATASETS_DIR, '{}_{}_{}.pickle'.format(naming, count, st_type))
