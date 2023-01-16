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

    root_mean_square_error = get_rmse(actual_table, predication_table)
    mean_based = get_rmse(actual_table, benchmark_table)

    return root_mean_square_error, mean_based


def get_best_k_recomnended_items(cf, k):
    # calculate the mean of each item column in the user_item_matrix
    items_means = cf.user_item_matrix.mean(skipna=True)
    # sort the items by their mean in descending order
    items_means = items_means.sort_values(ascending=False)
    # get the first k items
    top_k_items = items_means.head(k).index.tolist()
    return top_k_items


def precision_at_k(test_set, cf, k):
    # Create a pivot table of the test set to make it easier to access user-item ratings
    actual_table = pd.pivot_table(test_set, index='UserId', columns='ProductId', values='Rating')
    precision_k = []
    precision_k_benchmark = []
    # Get the top k recommended items from the benchmark method
    top_k_recommended_benchmark = get_best_k_recomnended_items(cf, k)

    for user_id, user_actual_rate in actual_table.iterrows():
        # Get the top k recommended items for the current user
        top_k_recommended = cf.recommend_items(user_id, k)
        # Get the items that have a rating of 3 or higher for the current user
        relevant_items = [col for col, rate in user_actual_rate.items() if rate >= 3]

        # If there are no relevant items, skip this user
        if not relevant_items:
            continue

        # Find the items that were recommended and are also relevant
        relevant_items_recommended = list(set(top_k_recommended) & set(relevant_items))
        relevant_items_recommended_benchmark = list(set(top_k_recommended_benchmark) & set(relevant_items))

        # Calculate the precision at k for the current user and append it to the list
        precision_k.append(len(relevant_items_recommended) / k)
        precision_k_benchmark.append(len(relevant_items_recommended_benchmark) / k)

    # Calculate the mean precision at k for all users and round to 5 decimal places
    return round(np.mean(precision_k), 5), round(np.mean(precision_k_benchmark), 5)


def recall_at_k(test_set, cf, k):
    actual_table = pd.pivot_table(test_set, index='UserId',
                                          columns='ProductId', values='Rating')

    return 0


