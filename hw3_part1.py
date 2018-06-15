from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#import Timer

# Generate the 5 files for init.


#Create Train and Test Files
"""
dataset = shuffle(pd.read_csv('dataset.csv')) #Randomly shuffle the rows
cols_to_x = ['user','movie']
cols_to_y = ['rate']
train_percent = 0.8
train_length = int(train_percent*dataset.shape[0])
test_length = train_length+1

train_x = (dataset[cols_to_x][0:train_length])
test_x = (dataset[cols_to_x][test_length:])
train_x.to_csv("train_x.csv")
test_x.to_csv("test_x.csv")

train_y = (dataset[cols_to_y][0:train_length])
test_y = (dataset[cols_to_y][test_length:])
train_y.to_csv("train_y.csv")
test_y.to_csv("test_y.csv")
"""

class ModelData:
    """The class reads 5 files as specified in the init function, it creates basic containers for the data.
    See the get functions for the possible options, it also creates and stores a unique index for each user and movie
    """


    def __init__(self, train_x, train_y, test_x, test_y, movie_data):
        """Expects 4 data set files with index column (train and test) and 1 income + genres file without index col"""

        self.train_x = pd.read_csv(train_x, index_col=[0])
        self.train_y = pd.read_csv(train_y, index_col=[0])
        self.test_x = pd.read_csv(test_x, index_col=[0])
        self.test_y = pd.read_csv(test_y, index_col=[0])
        self.movies_data = pd.read_csv(movie_data)
        self.users = self._generate_users()
        self.movies = self._generate_movies()
        self.incomes = self._generate_income_dict()
        self.movie_index = self._generate_movie_index()
        self.user_index = self._generate_user_index()

    def _generate_users(self):
        users = sorted(set(self.train_x['user']))
        return tuple(users)

    def _generate_movies(self):
        movies = sorted(set(self.train_x['movie']))
        return tuple(movies)

    def _generate_income_dict(self):
        income_dict = defaultdict(float)
        average_income = np.float64(self.movies_data['income'].mean())
        movies = sorted(set(self.movies_data['movie']))
        for m in movies:
            movie_income = int(self.movies_data[self.movies_data['movie'] == m]['income'])
            income_diff = movie_income * 10e-10
            income_dict[m] = income_diff
        return income_dict

    def _generate_user_index(self):
        user_index = defaultdict(int)
        for i, u in enumerate(self.users):
            user_index[u] = i
        return user_index

    def _generate_movie_index(self):
        movie_index = defaultdict(int)
        for i, m in enumerate(self.movies, start=len(self.users)):
            movie_index[m] = i
        return movie_index

    def get_users(self):
        """:rtype tuples of all users"""
        return self.users

    def get_movies(self):
        """:rtype tuples of all movies"""
        return self.movies

    def get_movie_index(self, movie):
        """:rtype returns the index of the movie if it exists or None"""
        return self.movie_index.get(movie, None)

    def get_user_index(self, user):
        """:rtype returns the index of the user if it exists or None"""
        return self.user_index.get(user, None)

    def get_movie_income(self, movie):
        """:rtype returns the income of the movie if it exists or None"""
        if self.incomes.get(movie, None) is None:
            print(movie)
        return self.incomes.get(movie, None)

    def get_movies_for_user(self, user):
        return self.train_x[self.train_x['user'] == user]['movie'].values

    def get_users_for_movie(self, movie):
        return self.train_x[self.train_x['movie'] == movie]['user'].values


def create_coefficient_matrix(train_x, data: ModelData = None):
    #matrix_timer = Timer.Timer('Matrix A creation')
    # TODO: Modify this function to return the coefficient matrix A as seen in the lecture (slides 24 - 37).
    users = data.get_users()
    matrix_a = np.array([[1 for _ in users] for _ in range(1000)])
    #matrix_timer.stop()
    return matrix_a


def create_coefficient_matrix_with_income(train_x, data: ModelData = None):
    #matrix_timer = Timer.Timer('Matrix A with income creation')
    # TODO: Modify this function to return a coefficient matrix A for the new model with income
    users = data.get_users()
    matrix_a = np.array([[1 for _ in users] for _ in range(1000)])
    #matrix_timer.stop()
    return matrix_a


def construct_rating_vector(train_y, r_avg):
    # TODO: Modify this function to return vector C as seen in the lecture (slides 24 - 37).
    y = [x for x in train_y.values]
    return np.array(y)


def fit_parameters(matrix_a, vector_c):
    # TODO: Modify this function to return vector b*, the solution of the equation (slides 24 - 37).
    result = np.ones(100)
    return result


def calc_parameters(r_avg, train_x, train_y, data: ModelData = None):
    # TODO: Modify this function to return the calculated average parameters vector b (slides 24 - 37).

    users = data.get_users()
    movies = data.get_movies()
    b = np.array(users + movies)
    print(b)
    return b


def calc_average_rating(train_y):
    # TODO: Modify this function to return the average rating r_avg.
    # DR: Done.
    r_avg = train_y["rate"].mean()
    return r_avg


def model_inference(test_x, vector_b, r_avg, data: ModelData = None):
    # TODO: Modify this function to return the predictions list ordered by the same index as in argument test_x
    predictions_list = []
    for i in test_x.index:
        # print(i)
        predictions_list += [r_avg]
    return predictions_list


def model_inference_with_income(test_x, vector_b, r_avg, data: ModelData = None):
    # TODO: Modify this function to return the predictions list ordered by the same index as in argument test_x
    # TODO: based on the modified model with income
    predictions_list = []
    for i in test_x.index:
        # print(i)
        predictions_list += [r_avg + data.get_movie_income(57)]
    return predictions_list


def calc_error(predictions_df, test_df):
    # TODO: Modify this function to return the RMSE
    return 1.75


def calc_avg_error(predictions_df, test_df):
    # TODO: Modify this function to return a dictionary of tuples {MOVIE_ID:(RMSE, RATINGS)}
    m_error = defaultdict(tuple)
    # In this example movie 3 was ranked by 5 users and has an RMSE of 0.9
    m_error[3] = (0.9, 5)
    return m_error


def plot_error_per_movie(movie_error):
    # TODO: Modify this function to plot the graph y:(RMSE of movie i) x:(number of users rankings)
    pass
