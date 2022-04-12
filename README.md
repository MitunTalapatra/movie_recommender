# Movie Recommender
This project aimed to develop a simple recommender system using user ratings and similairty analysis. Movielens dataset for 10,000 movies were used for this recommender functions and the recommender system is deployed in streamlit. 
Here is the link: https://share.streamlit.io/mituntalapatra/movie_recommender/main/streamlit.py

There are four types of recommendation given:
1. Random Selection -- randomly selected movies from the not seen movies by the user.
2. Most Popular -- popularity of the movies were calculated from the count of highest ratings other users. Every time user search, the app will suggest different popular movies.
3. Similarity 1 -- similarity is calculated by KNN clustering of all movies, detecting the clusters of the user's watched movies and recommend 3 movies from each clusters.
4. Similarity 2 -- NMF (Non-negative Matrix Factorization) was applied to calculate similarity and recommend randomly from top scored movies.
