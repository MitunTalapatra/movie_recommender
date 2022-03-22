from recommender import recommend_most_popular, recommend_random, recommend_from_same_cluster, recommend_with_NMF
from utils import lookup_movieId, match_movie_title, get_popularity, unseen_movies
import pandas as pd
import numpy as np
# example input of web application
user_rating = {
    'the lion king': 5,
    'terminator': 5,
    'star wars': 2
}
movies = pd.read_csv('data/movies_clusters_ratings.csv', index_col='movieid')  
user_mat = pd.read_csv('data/cleaned_user_item_matrix.csv', index_col=[0])
dictionary = pd.read_csv('data/cleaned_movies_dictionary.csv', index_col=[0])

# Please make sure that you output the ids and then modify the lookupmovieId to give the user the titles

### Terminal recommender:

print('>>>> Here are some movie recommendations for you:')
print('')
print(recommend_random(movies, user_rating, k=5))


print('')

print(recommend_most_popular(user_rating, movies, k=5))


print('')

print(recommend_from_same_cluster(user_rating, movies, k=3))

print('')

print(recommend_with_NMF(user_mat, user_rating, dictionary, k=5))



