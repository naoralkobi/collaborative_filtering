# Naor Alkobi 315679985
import collections
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def watch_data_info(data):

    # This function returns the first 5 rows for the object based on position.
    # It is useful for quickly testing if your object has the right type of data in it.
    print(data.head())

    # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
    print(data.info())

    # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
    print(data.describe(include='all').transpose())


def print_data(data):

    data_dict = data.to_dict(orient='list')

    users = data_dict.get('UserId')
    products = data_dict.get('ProductId')
    rating = data_dict.get('Rating')

    user_item_matrix_raw = pd.pivot_table(data, index='UserId',
                                          columns='ProductId', values='Rating', aggfunc=np.sum)

    counts_rating = user_item_matrix_raw.count()

    counts_products = user_item_matrix_raw.count(axis=1)

    print(f"number of users are :  {len(set(users))}")
    print(f"number of products ranked are : {len(set(products))}")
    print(f"number of ranking are: {len(rating)}")
    print(f"minimum number of ratings given to a product : {counts_rating.min()}")
    print(f"maximum number of ratings given to a product : {counts_rating.max()}")
    print(f"minimum number of products ratings by user : {counts_products.min()}")
    print(f"maximum number of products ratings by user : {counts_products.max()}")



