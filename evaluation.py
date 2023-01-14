# Naor Alkobi 315679985
import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_squared_error


def get_rmse(actual_table, predication_table):
    # Create a difference table
    diff_table = actual_table - predication_table

    # Square the difference table
    diff_table_squared = diff_table ** 2

    # Find the mean of the difference_squared table
    mean_difference_squared = diff_table_squared.mean().mean()

    # Take the square root of the mean
    rmse = np.sqrt(mean_difference_squared).round(5)

    return rmse


def RMSE(test_set, cf):
    actual_table = pd.pivot_table(test_set, index='UserId',
                                          columns='ProductId', values='Rating')
    predication_table = cf.pred

    mean_table = cf.user_item_matrix.copy()
    benchmark_table = cf.user_item_matrix.copy()
    mean_table = mean_table.apply(lambda x: x.mean(), axis=1)

    i = 0
    for index, series in benchmark_table.iterrows():
        new_value = mean_table[i]
        series = new_value
        benchmark_table.loc[index] = series
        i += 1

    rmse = get_rmse(actual_table, predication_table)
    mean_based = get_rmse(actual_table, benchmark_table)

    return rmse, mean_based


def precision_at_k(test_set, cf, k):
    "*** YOUR CODE HERE ***"
    pass

def recall_at_k(test_set, cf, k):
    "*** YOUR CODE HERE ***"
    pass
