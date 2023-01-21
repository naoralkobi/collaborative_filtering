# Naor Alkobi 315679985
import numpy as np
from sklearn.metrics import mean_squared_error


def RMSE(test_set, cf):

    # get the ratings from the test_set as a list
    test_ratings = test_set['Rating'].tolist()
    # use list comprehension to get the predicted ratings for each user-product pair in the test set
    pred_ratings = [cf.pred.loc[row['UserId'], row['ProductId']] for _, row in test_set.iterrows()]
    # use list comprehension to get the mean rating for each user in the test set
    mean_ratings = [np.nanmean(cf.user_item_matrix.loc[row['UserId']]) for _, row in test_set.iterrows()]

    # calculate the RMSE for the predicted ratings and the mean ratings
    cf_rmse = round(mean_squared_error(test_ratings, pred_ratings, squared=False), 5)
    mean_rmse = round(mean_squared_error(test_ratings, mean_ratings, squared=False), 5)

    # print the RMSE for the predicted ratings and the mean ratings
    if cf.strategy == 'user':
        print(f"user-based CF RMSE {cf_rmse}")
    else:
        print(f"item-based CF RMSE {cf_rmse}")
    print(f"mean based (benchmark) RMSE {mean_rmse}")


def get_best_k_recomnended_items(train_data, k):
    """
        This function returns the top k products based on their mean rating in the train_data
    """
    mean_user_rating = train_data.mean(axis=0).sort_values(ascending=False)[:k].index
    return mean_user_rating.tolist()


def precision_at_k(test_set, cf, k):
    """
        This function calculates the precision at k for a given test set, recommendation model object and k.
        The test set is a DataFrame that contains the actual ratings, and the recommendation model object (cf)
        is used to get the recommended items for each user.
    """
    bench = get_best_k_recomnended_items(cf.user_item_matrix, k)
    list_of_precision = []
    list_of_precision_bench = []
    dt = test_set.groupby('UserId')
    for user_id, rows in dt:
        recommended_k = cf.recommend_items(user_id, k)
        relevant_products = rows[rows['Rating'] >= 3]['ProductId'].tolist()
        if len(relevant_products) == 0:
            continue
        intersection = set(relevant_products).intersection(set(recommended_k))
        intersection_bench = set(relevant_products).intersection(set(bench))
        precision = len(intersection) / k
        list_of_precision.append(precision)
        precision_bench = len(intersection_bench) / k
        list_of_precision_bench.append(precision_bench)

    print(f"precision at {k} is {np.round(np.mean(list_of_precision), 5)}")
    print(f"precision at {k} (benchmark) is {np.round(np.mean(list_of_precision_bench), 5)}")


def recall_at_k(test_set, cf, k):
    bench = get_best_k_recomnended_items(cf.user_item_matrix, k)
    list_of_recall = []
    list_of_recall_bench = []
    for user_id in set(test_set['UserId']):
        recommended_k = cf.recommend_items(user_id, k)
        relevant_products = test_set[(test_set['UserId'] == user_id) & (test_set['Rating'] >= 3)]['ProductId'].tolist()
        if not relevant_products:
            continue
        # Get the intersection of recommended items and relevant items
        intersection = set(relevant_products).intersection(set(recommended_k))
        # Get the intersection of the best items and relevant items
        intersection_bench = set(relevant_products).intersection(set(bench))
        recall = len(intersection) / len(relevant_products)
        list_of_recall.append(recall)
        recall_bench = len(intersection_bench) / len(relevant_products)
        list_of_recall_bench.append(recall_bench)

    print(f"recall at {k} is {np.round(np.mean(list_of_recall), 5)}")
    print(f"recall at {k} (benchmark) is {np.round(np.mean(list_of_recall_bench), 5)}")


