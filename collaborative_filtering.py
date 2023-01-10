# Naor Alkobi 315679985
import collections
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy='user'):
        self.strategy = strategy
        self.similarity = np.NaN

    def fit(self, matrix):
        " * ** YOUR CODE HERE ** * "

        matrix = matrix.apply(lambda x: round(x - matrix.mean(axis=1), 2))

        self.user_item_matrix = matrix

        if self.strategy == 'user':
            # User - User based collaborative filtering
            start_time = time.time()

            self.pred = pd.DataFrame() #self.pred should contain your prediction metrix.


            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))

            return self


        elif self.strategy == 'item':
            # Item - Item based collaborative filtering
            start_time = time.time()

            self.pred = pd.DataFrame() #self.pred should contain your prediction metrix.

            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))

            return self

    def recommend_items(self, user_id, k=5):


        " * ** YOUR CODE HERE ** * "
        if self.strategy == 'user':

            # Fill missing values with zero of each column
            self.user_item_matrix.fillna(0, inplace=True)

            # Create a similarity matrix
            item_similarity = cosine_similarity(self.user_item_matrix.T)

            user_ratings = self.user_item_matrix.loc[user_id]

            user_index = user_ratings.index
            similar_items = item_similarity[user_index]
            similar_items = similar_items.T[user_index]
            similar_items = similar_items.sum(axis=1)
            similar_items = similar_items.sort_values(ascending=False)

            top_k_items = similar_items.index[:k]
            return top_k_items

        elif self.strategy == 'item':
            return None

