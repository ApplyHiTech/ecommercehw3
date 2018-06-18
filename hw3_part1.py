from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix

from sklearn.utils import shuffle

import Timer

# Generate the 5 files for init.


# Create Train and Test Files
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
        print(self.movie_index)
        print("user_index")
        print(self.user_index)

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
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html#scipy.sparse.dok_matrix

    # TO DO: Modify this function to return the coefficient matrix A as seen in the lecture (slides 24 - 37).

    #matrix_timer.stop()
    total_users = len(data.user_index)
    column_len = total_users+len(data.movie_index)

    row_len = train_x.shape[0]

    print("S MATRIX")
    print(total_users, row_len, column_len)

    # Matrix looks as follows:
    #
    # <columns = features, equal the index of users, u1,u2,u3...u_n followed by index of movies, m1,m2,m3...m_n>
    # <rows = 1 sample in the train_x file>
    #

    S = dok_matrix((row_len + 1, column_len), dtype=np.float32)
    print(S)
    for j in range(column_len):
        S[0, j] = 0.001 * np.random.random()
    i=0 # the row in the sparse matrix.

    for index,row in data.train_x.iterrows():
        i+=1
        # for each row in train_x, add 1 to user index and 1 to movie index
        user = row["user"]
        movie = row["movie"]
        u_index = data.user_index[user]
        m_index = data.movie_index[movie]

        S[i,u_index]=1
        S[i,m_index]=1


    return S


def create_coefficient_matrix_with_income(train_x, data: ModelData = None):
    #matrix_timer = Timer.Timer('Matrix A with income creation')
    # TODO: Modify this function to return a coefficient matrix A for the new model with income
    users = data.get_users()
    matrix_a = np.array([[1 for _ in users] for _ in range(1000)])
    #matrix_timer.stop()
    return matrix_a


def construct_rating_vector(train_y, r_avg):
    # TO DO: Modify this function to return vector C as seen in the lecture (slides 24 - 37).
    # DR done, just subtracted r_avg from each value
    y = [x-r_avg for x in train_y.values]
    return np.array(y)


def fit_parameters(matrix_a, vector_c):
    # TODO: Modify this function to return vector b*, the solution of the equation (slides 24 - 37).
    result = np.ones(100)
    return result


def calc_parameters(r_avg, train_x, train_y, data: ModelData = None):
    # TO DO: Modify this function to return the calculated average parameters vector b (slides 24 - 37).

    users = data.get_users()
    movies = data.get_movies()
    print("There are %s users and %s movies" % (len(users),len(movies)))
    b_user = {} # users
    b_item = {} # movie
    b0 = []
    for u in users:
        # get index of all users - movie pairings
        list_indexes = data.train_x[data.train_x["user"]==u].index.tolist()
        tot_rating = 0
        for j in list_indexes:
            tot_rating += data.train_y.loc[j]["rate"]
        b_user[u]= tot_rating/len(list_indexes) - r_avg
        b0 += [[u,tot_rating/len(list_indexes) - r_avg]]
    #print(b_user)

    for m in movies:
        # get index of all movies - movie pairings
        list_indexes = data.train_x[data.train_x["movie"]==m].index.tolist()
        tot_rating = 0
        for j in list_indexes:
            tot_rating += data.train_y.loc[j]["rate"]
        b_item[m]= tot_rating/len(list_indexes) - r_avg
        b0 += [[m, tot_rating / len(list_indexes) - r_avg]]
    return b0


def calc_average_rating(train_y):
    # TO DO: Modify this function to return the average rating r_avg.
    # DR: Done.
    r_avg = train_y["rate"].mean()
    return r_avg


def model_inference(test_x, vector_b, r_avg, data: ModelData = None):
    # TO DO: Modify this function to return the predictions list ordered by the same index as in argument test_x
    # DR Done... though messy Vector to Dictionary conversion.
    users = data.get_users()
    movies = data.get_movies()

    # from vector to dict
    b_user = {}
    b_movies = {}
    for i in range(len(users)):
        k,v=vector_b[i]
        b_user[k]=v

    for j in range(len(movies)):
        k,v=vector_b[j+len(users)]
        b_movies[k]=v

    # Testing to make sure the lengths are equivalent.
    if len(vector_b)==len(users)+len(movies):
        #print("great! ")
        pass
    else:
        print("Bad")

    # Generate Prediction List
    predictions_list = []
    for index,row in test_x.iterrows():
        user = row["user"]
        movie = row["movie"]
        bu=0
        bi=0
        if user in b_user.keys():
            bu = b_user[user]
        if bi in b_movies.keys():
            bi = b_movies[movie]

        predictions_list += [r_avg+bu+bi]


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
    # TO DO: Modify this function to return the RMSE
    #
    val = ((predictions_df.r_hat-test_df.rate)**2).mean()**.5
    return val


def calc_avg_error(predictions_df, test_df):
    # TO DO: Modify this function to return a dictionary of tuples {MOVIE_ID:(RMSE, RATINGS)}
    # DR Done
    m_error = defaultdict(tuple)

    # Get unique movie ids
    movies = predictions_df.movie.unique()
    for m in movies:
        df_predict = predictions_df[predictions_df["movie"]==m]
        index_list = df_predict.index.tolist()
        df_test = test_df.loc[index_list]
        rmse = calc_error(df_predict,df_test)
        numb_ratings = len(index_list)
        m_error[m]=(rmse,numb_ratings)

    return m_error


def plot_error_per_movie(movie_error):
    # TO DO: Modify this function to plot the graph y:(RMSE of movie i) x:(number of users rankings)
    # DR done.

    x = []
    y = []
    for k,v in movie_error.items():
        x += [v[1]]
        y += [v[0]]

    plt.scatter(x,y)
    plt.xlabel("Number of Users")
    plt.ylabel("RMSE")
    plt.show()
    pass
