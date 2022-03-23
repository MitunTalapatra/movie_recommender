"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import lookup_movieId, match_movie_title, get_popularity, unseen_movies, get_cluster
from sklearn.impute import SimpleImputer
from sklearn.decomposition import NMF

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

movies = pd.read_csv('data/movies_clusters_ratings.csv', index_col='movieid')  
user_mat = pd.read_csv('data/cleaned_user_item_matrix.csv', index_col=[0])
dictionary = pd.read_csv('data/cleaned_movies_dictionary.csv', index_col=[0])



def recommend_random(movies, user_rating, k=5):
    """
    return k random unseen movies for user 
    """
    #recommend = unseen_movies(movies, user_rating)
    random_movies = np.random.choice(list(recommend.index), replace=False, size=k)
    #random_movies = np.random.choice(list(movies['title']), replace=False, size=k)
    
    return random_movies


def recommend_most_popular(user_rating, movies, k=5):
    """
    return k movies from list of 50 best rated movies unseen for user
    """
    popularity = get_popularity(user_mat, dictionary)
    movies = movies.join(popularity[['null_count','total_top_rating']])
    recommend = unseen_movies(movies, user_rating)

    most_popular_100 = recommend.sort_values('total_top_rating', ascending=False).head(100)
    most_popular = list(most_popular_100.sample(frac = 1.0).head(5).index)
    return most_popular


def recommend_from_same_cluster(user_rating, movies, k=3):
    """
    Return k most similar movies to the one spicified in the movieID
    
    INPUT
    - user_rating: a dictionary of titles and ratings
    - movies: a data frame with movie titles and cluster number
    - k: number of movies to recommend

    OUTPUT
    - title: the matched movie title (with fuzzy wuzzy) of the best ranked entry
    - movie_titles 
    """
    user_cluster = get_cluster(movies, user_rating)
    popularity = get_popularity(user_mat, dictionary)
    movies = movies.join(popularity[['null_count','total_top_rating']])
    unseen = unseen_movies(movies, user_rating)
    movies_same_cluster = unseen.loc[unseen.apply(lambda x: x.cluster_no in user_cluster, axis=1)]
    
    top_20 = movies_same_cluster.sort_values('total_top_rating',ascending =False).groupby('cluster_no').head(20)
    recommend_2 = top_20.sample(frac = 1.0).groupby('cluster_no').head(3)
    recommend_cluster = list(recommend_2.index.values)
            
    return recommend_cluster



def recommend_with_NMF(user_mat, user_rating, dictionary, k=5):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of movieIds
    """
    # process user-item matrix
    user_r = user_mat.T.reset_index()
    user_r['movieid'] = user_r['index'].astype(int)
    user_r.set_index('movieid', inplace=True)
    user_title = user_r.join(dictionary)
    user_title.set_index('title', inplace=True)
    user_title.drop('index', axis=1, inplace=True)

    # syn user rating with user-item matrix
    user_df = pd.DataFrame(user_rating, index=[0])
    user_t = user_df.T.reset_index()
    user_movie_entries = list(user_t['index'])
    movie_titles = list(user_title.index.values)
    parsed_title = [match_movie_title(title, movie_titles) for title in user_movie_entries]
    user_df.columns = parsed_title
    user_df = user_df.T
    user_df.rename_axis(index='title', inplace=True)

    # remove null values
    all_user_info = user_title.join(user_df)
    user = all_user_info[0]
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(all_user_info.T)
    impute_transform = imputer.transform(all_user_info.T)
    new_df = pd.DataFrame(impute_transform.T, columns=all_user_info.columns, index=all_user_info.index.values)

    # run nmf
    nmf = NMF(n_components=5, max_iter=1000)
    nmf.fit(new_df)
    P = nmf.transform(new_df)
    Q = nmf.components_
    Q_df = pd.DataFrame(Q, columns=new_df.columns)
    PQ = P.dot(Q)
    PQ_df = pd.DataFrame(PQ, columns=new_df.columns, index=new_df.index.values)

    #recommendation
    mask = ~PQ_df.index.isin(user_df.index)
    result = PQ_df.loc[mask]
    top_50 = PQ_df[0].sort_values(ascending = False).head(50)
    recommend_5 = top_50.sample(frac = 1.0).head(5)
    recommend_nmf = list(recommend_5.index.values)
    
    return recommend_nmf

def recommend_with_user_similarity(user_item_matrix, user_rating, k=5):
    pass


def similar_movies(movieId, movie_movie_distance_matrix):
    pass

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
    print(recommend_from_same_cluster(user_rating, movies, k=3))