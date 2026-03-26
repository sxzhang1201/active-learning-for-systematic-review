import os
import pandas as pd
import numpy as np
import config


def correlation():
  inp = os.path.join(config.CONSOLIDATION_DIR, 'correlation_input.xlsx')
  out = os.path.join(config.CONSOLIDATION_DIR, 'correlation_output.xlsx')
  os.makedirs(config.CONSOLIDATION_DIR, exist_ok=True)

  df = pd.read_excel(inp)
  cov = df.corr()
  cov.to_excel(out, sheet_name='sheet1', index=False)