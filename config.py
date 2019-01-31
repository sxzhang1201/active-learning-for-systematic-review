import os

CWD = os.getcwd()

DB_FILE = CWD + '/Database/miner_database.db'

DEBUG = False

### Pubmed API config:
PUBMED_EMAIL = 's.x.zhang@amc.uva.nl'
DB_INSERT_LIMIT = 100

### CLEF specific config
STORE_LOCATION = CWD + '/Pubmed results/'
TRAIN_QREL_LOCATION = CWD + '/Database/CLEF_data/qrel_abs_train'
TEST_QREL_LOCATION = CWD + '/Database/CLEF_data/qrel_abs_test'
DOC_COLLECTION = CWD + '/Text_data/document_collection.pickle'

TEXT_DATA_LOCATION = CWD + '/Text_data'

TF_MATRIX = CWD + '/Text_data/tf_matrix.pickle'
MATRIX_COLLECTION = CWD + '/Text_data/matrix_collection.pickle'

RESULTS_LOCATION = CWD + '/Results'

EXTRA_STOPWORDS = []
KEEP_DASHES = False
KEEP_NUMBERS = False

NUM_CORES = 1
NUMBER_CV = 3
NUM_ITERATIONS = 2

SIMILAR_SIZES = [1, 5, 10]



# The following configuration parameters remains to be updated.


DOC_TERM_MATRIX = CWD + '/Database/term_document_matrix.pickle'

# Figure:
WORD_FREQUENCY = CWD + '/Figure/term_frequency.csv'

# 2. Baseline model design
# Result Objects:
GRID_SEARCH = CWD + '/Consolidation/grid_search.pickle'
TF_IDF_RESULTS = CWD + '/Consolidation/tf_idf_test.pickle'
CLASS_IMBALANCE_RESULTS = CWD + '/Consolidation/class_imbalance.pickle'


# Figure:
HEAT_MAP = CWD + '/Figure/heatmap.pdf'
TF_IDF_FIGURE = CWD + '/Figure/tf_idf.pdf'
CLASS_IMBALANCE_FIGURE = CWD + '/Figure/class_imbalance.pdf'

# 3. Active learning model design

# Figure:
QUERY_STRATEGY_FIGURE = CWD + '/Figure/query_strategy.pdf'
DOCUMENT_WEIGHT = CWD + '/Figure/importance.pdf'
RELEVANT_DOCUMENT_WEIGHT = CWD + '/Figure/relevant_importance.pdf'