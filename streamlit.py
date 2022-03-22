#import std libraries
import pandas as pd
import numpy as np
from utils import lookup_movieId, match_movie_title, get_popularity, unseen_movies, get_cluster
from sklearn.impute import SimpleImputer
from sklearn.decomposition import NMF
from recommender import recommend_random, recommend_most_popular, recommend_from_same_cluster, recommend_with_NMF

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import streamlit as st 
import seaborn as sns
import plotly.express as px

movies = pd.read_csv('data/movies_clusters_ratings.csv', index_col='movieid')  
user_mat = pd.read_csv('data/cleaned_user_item_matrix.csv', index_col=[0])
dictionary = pd.read_csv('data/cleaned_movies_dictionary.csv', index_col=[0])

#choice = st.sidebar.radio('Hope this was intersting',['yes','no'])
#if choice == 'yes':
    
# Write a title
st.title('Movie Recommender')
# Write data taken from https://allisonhorst.github.io/palmerpenguins/
st.write('**Little** *app* for finding a good movie to watch [datset](https://grouplens.org/datasets/movielens/)')
# Put image https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/man/figures/lter_penguins.png
st.image('https://images.app.goo.gl/Qt3ziMxz2GxzvmbB9')
st.write('Please enter 3 movies you have watched and rate them from 0-5')

input_1 = st.text_input('Movie 1', '') 
rating_1= st.slider('Rate movie 1', 0.0, 5.0, (0.0, 5.0))

input_2 = st.text_input('Movie 2', '') 
rating_2= st.slider('Rate movie 2', 0.0, 5.0, (0.0, 5.0))

input_3 = st.text_input('Movie 3', '') 
rating_3= st.slider('Rate movie 3', 0.0, 5.0, (0.0, 5.0))

user_rating = {
    input_1: rating_1,
    input_1: rating_2,
    input_1: rating_3
}
# Write heading for Random Recommendation
st.header('Random Selection')
st.write('Movies randomly selected for you...', recommend_random(movies, user_rating, k=5))

# Write heading for Popular Recommendation
st.header('Most popular')
st.write('Most popular movies for you...', recommend_most_popular(user_rating, movies, k=5))

# Write heading for Similar cluster Recommendation
st.header('Similarity 1')
st.write('Similar movies based on your experience...', recommend_from_same_cluster(user_rating, movies, k=3))

# Write heading for Similar cluster Recommendation
st.header('Similarity 1')
st.write('Similar movies based on other users experience...', recommend_with_NMF(user_mat, user_rating, dictionary, k=5))