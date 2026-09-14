"""Project 1 Gradescope submission.

Generated from the Project 1 notebook. This module intentionally contains no
notebook magics, grading calls, or top-level College.csv loading.
"""

import numpy as np
import pandas as pd

def read_data(input_file):
    return pd.read_csv(input_file)


def set_index_func(input_file):
    input_file.set_index('Names', inplace = True)
    return input_file


def get_college_least_tuition(df):
    college_least_tuition = df['Outstate'].idxmin()
    return college_least_tuition


def convert_to_dataframe_column(input_file):
    return input_file[['PhD']]


def retrieve_column_length(input_column):
    return len(input_column)


def retrieve_private_top10(input_file):
    return input_file[['Private', 'Top10perc']].iloc[[15, 16], :]


def retrieve_private_column(input_private_top10):
    return len(input_private_top10['Private'])


def locate_penn(input_file):
    return input_file.loc[['University of Pennsylvania']]


def retrieve_many_penns(input_file):
    return input_file[input_file.index.str.contains('Penn', regex=False, na=False)]
