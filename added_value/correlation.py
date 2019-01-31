import pandas as pd
import xlsxwriter
import numpy as np


def correlation():

  # Read the xlsx file that contains the variables to be analysed with each column representing one variable.
  df = pd.read_excel('../Consolidation/correlation_input.xlsx')

  # Perform Pearson coefficents analysis
  cov = df.corr()

  # Export the results
  cov.to_excel('../Consolidation/correlation_output.xlsx', sheet_name='sheet1', index=False)