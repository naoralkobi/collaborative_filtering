# Naor Alkobi 315679985
import collections
import heapq
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy='user'):
        self.pred_zeros = None
        self.pred = None
        self.user_item_matrix = None
        self.strategy = strategy

    def fit(self, matrix):

        self.user_item_matrix = matrix

        # Assign the input matrix to an instance variable for future use
        self.user_item_matrix = matrix
        # Calculate the mean rating for each user
        mean_user_rating = np.nanmean(matrix, axis=1).reshape(-1, 1)
        # Calculate the difference between each rating and the mean rating for that user
        ratings_diff = (matrix - mean_user_rating + 0.001)

        # Replace NaN values with 0
        ratings_diff[np.isnan(ratings_diff)] = 0

        if self.strategy == 'user':
            # User - User based collaborative filtering
            start_time = time.time()

            user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
            pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
                [np.abs(user_similarity).sum(axis=1)]).T
            pred = np.round(pred, 2)
            self.pred = pd.DataFrame(pred, index=self.user_item_matrix.index, columns=self.user_item_matrix.columns)
            self.pred_zeros = self.pred.where(self.user_item_matrix.isna(), 0)

            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))

            return self

        elif self.strategy == 'item':
            # Item - Item based collaborative filtering
            start_time = time.time()

            item_similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')
            pred_val = mean_user_rating + ratings_diff.dot(item_similarity) / np.array(
                [np.abs(item_similarity).sum(axis=1)])
            pred_val = np.round(pred_val, 2)
            pred_val.index = self.user_item_matrix.index
            pred_val.columns = self.user_item_matrix.columns
            self.pred = pred_val
            self.pred_zeros = self.pred.where(self.user_item_matrix.isna(), 0)

            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))

            return self

    def recommend_items(self, user_id, k=5):
        if user_id not in self.user_item_matrix.index:
            return None
        top_k = self.pred_zeros.loc[user_id].sort_values(kind='mergesort', ascending=False)[:k]
        top_k_list = top_k.index.values
        return top_k_list
