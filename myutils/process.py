# data processing

import csv
import pandas as pd
from collections import Counter


def PropCounter(x, round_to=3):
    x_counter = Counter(x)
    return {i: round(j / sum(x_counter.values()), round_to) for i, j in x_counter.items()}


def flatten_list(ls):
    return [item for row in ls for item in row]


def get_list_from_csv(file, vartype='str'):
    elements = []
    with open(file) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            try:
                if vartype == 'str':
                    element = str(row[0])
                elif vartype == 'int':
                    element = int(row[0])
                elif vartype == 'float':
                    element = float(row[0])
                else:
                    raise ValueError('Paramter vartype needs to be one of "str", "int", or "float".')
                elements.append(element)
            except ValueError:
                pass
    return elements


def identify_elements_in_dataframe_column(dataframe, column_name, x):
    """
    Identify elements in a column of a pandas DataFrame that occur in fewer than 'x' rows.

    Args:
        dataframe: The pandas DataFrame.
        column_name: The name of the column to analyze.
        x: The threshold value for the number of occurrences.

    Returns:
        A list of elements that occur in fewer than 'x' rows in the specified column.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input 'dataframe' must be a pandas DataFrame.")

    if column_name not in dataframe.columns:
        raise ValueError("Column name not found in the DataFrame.")

    element_counts = dataframe[column_name].value_counts()
    infrequent_elements = element_counts[element_counts < x].index.tolist()

    return infrequent_elements
