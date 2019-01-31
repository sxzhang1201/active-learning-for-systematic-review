#1. Data Preparation
STEMMER_TYPE = 'porter'

#2. Baseline Model Design
OPTIMAL_ALPHA = 1e-6
OPTIMAL_TOLERANCE = 1e-3
STANDARD_TYPE = 'tf_idf'
CLASS_IMBALANCE = 'random_under'

#3. Active Learning Model Design
QUERY_STRATEGY = 'least_confidence'  # uncertainty sampling
SAMPLE_WEIGHT = 1