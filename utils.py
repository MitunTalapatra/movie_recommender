"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np
from fuzzywuzzy import process


movies = pd.read_csv('data/movies_clusters_ratings.csv', index_col='movieid')  
user_mat = pd.read_csv('data/cleaned_user_item_matrix.csv', index_col=[0])
dictionary = pd.read_csv('data/cleaned_movies_dictionary.csv', index_col=[0])

def unseen_movies(movies, user_rating):
    """
    return list of all unseen movies by selecting the movies without rating by user 
    """
    user = pd.DataFrame(user_rating, index=[0])
    user_t = user.T.reset_index()
    user_movie_entries = list(user_t["index"])
    movie_titles = list(movies["title"])
    parsed_title = [match_movie_title(title, movie_titles) for title in user_movie_entries]
    
    unseen_movies = movies.copy()
    unseen_movies = unseen_movies.reset_index()
    unseen_movies = unseen_movies.set_index("title")
    unseen_movies.drop(parsed_title, inplace=True)
    
    return unseen_movies

def match_movie_title(input_title, movie_titles):
    """
    Matches inputed movie title to existing one in the list with fuzzywuzzy
    """
    matched_title = process.extractOne(input_title, movie_titles)[0]

    return matched_title

def print_movie_titles(movie_titles):
    """
    Prints list of movie titles in cli app
    """    
    for movie_id in movie_titles:
        print(f'            > {movie_id}')
    pass


def create_user_vector(user_rating, movies):
    """
    Convert dict of user_ratings to a user_vector
    """       
    # generate the user vector
    print(user_rating)
    user_vector = None
    return user_vector


def lookup_movieId(movies, movieId):
    """
    Convert output of recommendation to movie title
    """
    # match movieId to title
    movies = movies.reset_index()
    boolean = movies["movieid"] == movieId
    movie_title = list(movies[boolean]["title"])[0]
    return movie_title

    return movie_title

def get_popularity(user_mat, dictionary):
    user_mat = user_mat.T
    user_mat = user_mat.astype(float)
    
    null_count = []
    for i in user_mat.index:
        count = user_mat.loc[[i]].isna().sum().sum()
        null_count.append(count)
    
    user_mat['null_count'] = null_count
    user_mat['total_top_rating'] = user_mat.select_dtypes(np.number).gt(3.5).sum(axis=1)
    popular = user_mat[['null_count','total_top_rating']]
    popular = popular.reset_index()
    popular['movieid'] = popular['index'].astype(int)
    popular.set_index('movieid', inplace=True)
    movies_popularity = popular.join(dictionary)
    movies_popularity = movies_popularity.drop('index', axis=1)
    
    return movies_popularity

def get_cluster(movies, user_rating):
    """
    return list of all unseen movies by selecting the movies without rating by user 
    """
    user = pd.DataFrame(user_rating, index=[0])
    user_t = user.T.reset_index()
    user_movie_entries = list(user_t["index"])
    movie_titles = list(movies["title"])
    parsed_title = [match_movie_title(title, movie_titles) for title in user_movie_entries]
    user_cluster = list(movies['cluster_no'].loc[movies.apply(lambda x: x.title in parsed_title, axis=1)])
    
    return user_cluster

if __name__ == "__main__":
    user_rating = {
        "four rooms": 5,
        "sudden death": 3,
        "othello": 4,
        "nixon": 3,
        "Golden eye": 1,
        "total eclipse": 5,
        "nadja": 3
    }
    print(create_user_vector(user_rating, movies))