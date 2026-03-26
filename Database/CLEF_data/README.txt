Why this folder looks empty
-----------------------------
The CLEF relevance (qrel) files are NOT part of your code repository. The thesis used the
CLEF 2017 eHealth / TAR (Technologically Assisted Reviews) shared task data. You have to
download them once and put them here (or point config.py to wherever you saved them).

Easiest: download from the official CLEF-TAR GitHub mirror
------------------------------------------------------------
Base URL (raw files):
  https://raw.githubusercontent.com/CLEF-TAR/tar/master/

Training qrels (save AS qrel_abs_train in THIS folder):
  2017-TAR/training/qrels/qrel_abs_train

Test qrels — upstream file is named with .txt; save locally AS qrel_abs_test (no .txt)
to match config.py:
  2017-TAR/testing/qrels/qrel_abs_test.txt

Example (run from the project root, with curl):

  mkdir -p Database/CLEF_data
  curl -L -o Database/CLEF_data/qrel_abs_train \
    https://raw.githubusercontent.com/CLEF-TAR/tar/master/2017-TAR/training/qrels/qrel_abs_train
  curl -L -o Database/CLEF_data/qrel_abs_test \
    https://raw.githubusercontent.com/CLEF-TAR/tar/master/2017-TAR/testing/qrels/qrel_abs_test.txt

Then you should have:
  Database/CLEF_data/qrel_abs_train
  Database/CLEF_data/qrel_abs_test

Alternatively: clone https://github.com/CLEF-TAR/tar and copy those two files from
2017-TAR/training/qrels/ and 2017-TAR/testing/qrels/ under the names above.

Related repo (different layout, IR task assessments):
  https://github.com/CLEFeHealth/CLEFeHealth2017IRtask/tree/master/assessments/2017

Line format (whitespace-separated):
  review_id  q0  pmid  included
(where included is typically 0 or 1).
