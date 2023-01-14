# Naor Alkobi 315679985
import collections
import heapq
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy='user'):
        self.pred = None
        self.user_item_matrix = None
        self.strategy = strategy
        self.similarity = np.NaN
        self.products_id = []
        self.users_id = []
        self.pred_val = 0

    def create_users_and_products_list(self, matrix):
        for userId, productId in matrix.iterrows():
            self.users_id.append(userId)
        self.products_id.extend(matrix.head())

    def fit(self, matrix):

        self.create_users_and_products_list(matrix)
        self.user_item_matrix = matrix
        ratings = matrix.to_numpy()
        mean_user_rating = matrix.mean(axis=1).to_numpy().reshape(-1, 1)
        ratings_diff = (ratings - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0

        if self.strategy == 'user':
            # User - User based collaborative filtering
            start_time = time.time()

            user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')

            self.pred = self.user_item_matrix.copy()
            self.pred.values[:] = (mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
                [np.abs(user_similarity).sum(axis=1)]).T).round(2)

            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))

            return self

        elif self.strategy == 'item':
            # Item - Item based collaborative filtering
            start_time = time.time()

            item_similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')

            self.pred = self.user_item_matrix.copy()
            self.pred.values[:] = (mean_user_rating + ratings_diff.dot(item_similarity) / np.array(
                [np.abs(item_similarity).sum(axis=1)])).round(2)

            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))

            return self

    def recommend_items(self, user_id, k=5):

        if user_id not in self.users_id:
            print("Error in userID")
            return None

        if self.strategy == 'user':
            # get specific row values from predication matrix.d
            predicted_ratings_row = self.pred.loc[self.pred.index == user_id].copy()
            # get specific row values from user_item matrix.
            data_matrix_row = self.user_item_matrix.loc[self.user_item_matrix.index == user_id]

            predicted_ratings_row[~np.isnan(data_matrix_row)] = 0

            # get the first k.
            sorted_indices = np.flip(np.argsort(predicted_ratings_row.values))[0][:k].tolist()
            k_items = []
            for index in sorted_indices:
                rate = float(predicted_ratings_row[self.products_id[index]].values)
                self.pred_val = rate
                k_items.append(self.products_id[index])
            return k_items

        elif self.strategy == 'item':
            # get specific row values from predication matrix.d
            predicted_ratings_row = self.pred.loc[self.pred.index == user_id].copy()
            # get specific row values from user_item matrix.
            data_matrix_row = self.user_item_matrix.loc[self.user_item_matrix.index == user_id]

            predicted_ratings_row[~np.isnan(data_matrix_row)] = 0

            indices = np.flip(np.argsort(predicted_ratings_row.values))[0].tolist()
            values = collections.defaultdict(list)
            for index in indices:
                rate = float(predicted_ratings_row[self.products_id[index]].values)
                values[rate].append(index)

            smallest_indices = []
            remain = k
            for key in list(values.keys()):
                val = values[key]
                val.sort()
                if len(val) >= remain:
                    smallest_indices.extend(val[:remain])
                    break
                else:
                    smallest_indices.extend(val)
                    remain -= len(val)

            smallest_indices.sort()
            # get the first k.
            smallest_indices = smallest_indices[:k]

            k_items = []
            for index in smallest_indices:
                rate = float(predicted_ratings_row[self.products_id[index]].values)
                self.pred_val = rate
                k_items.append(self.products_id[index])
            return k_items

