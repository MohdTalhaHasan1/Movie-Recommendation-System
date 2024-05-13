# from flask import Flask, render_template, request, session
# import pandas as pd
# from surprise import Reader, Dataset, SVD

# # Load the data
# movies = pd.read_csv('movies.csv')
# ratings = pd.read_csv('ratings.csv')

# # Create the Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Set a secret key for the session

# # Create a reader and load the data into a surprise dataset
# reader = Reader(rating_scale=(0, 5))
# data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# # Train the recommendation model
# trainset = data.build_full_trainset()
# algo = SVD()
# algo.fit(trainset)

# # Function to get top-N recommendations for a user
# def get_recommendations(user_id, n=10):
#     user_ratings = ratings[ratings['userId'] == user_id]
#     movies_rated = user_ratings['movieId'].tolist()
    
#     # Get the movies not rated by the user
#     unrated_movies = movies[~movies['movieId'].isin(movies_rated)]['movieId']
    
#     # Predict ratings for unrated movies
#     predictions = [algo.predict(user_id, movie_id)[3] for movie_id in unrated_movies]
    
#     # Sort the predictions and get the top-N recommendations
#     top_n = sorted(zip(unrated_movies, predictions), key=lambda x: x[1], reverse=True)[:n]
    
#     # Get the movie titles and genres for the recommendations
#     recommendations = []
#     for movie_id, _ in top_n:
#         movie = movies[movies['movieId'] == movie_id].iloc[0]
#         recommendations.append({
#             'title': movie['title'],
#             'genres': movie['genres']
#         })
    
#     return recommendations

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         user_id = request.form['user_id']
#         session['user_id'] = user_id
#         return render_template('search.html')
#     return render_template('login.html')

# @app.route('/search', methods=['GET', 'POST'])
# def search():
#     if 'user_id' not in session:
#         return render_template('login.html')
    
#     if request.method == 'POST':
#         search_term = request.form['search_term']
#         recommendations = get_recommendations(session['user_id'])
#         filtered_recommendations = [rec for rec in recommendations if search_term.lower() in rec['title'].lower()]
#         return render_template('recommendations.html', recommendations=filtered_recommendations)
    
#     return render_template('search.html')

# if __name__ == '__main__':
#     app.run(debug=True)

































# from flask import Flask, render_template, request, session
# import pandas as pd
# from surprise import Reader, Dataset, SVD
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import os

# # Load the data
# movies = pd.read_csv('movies.csv')
# ratings = pd.read_csv('ratings.csv')

# # Create the Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # Set the template directory
# template_dir = os.path.join(os.path.dirname(__file__), 'templates')
# app.template_folder = template_dir

# # Create a reader and load the data into a surprise dataset
# reader = Reader(rating_scale=(0, 5))
# data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# # Train the recommendation model
# trainset = data.build_full_trainset()
# algo = SVD()
# algo.fit(trainset)

# # Preprocess movie titles for search
# movies['clean_title'] = movies['title'].str.lower().str.replace('[^a-zA-Z0-9\s]', '')
# vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# tfidf = vectorizer.fit_transform(movies["clean_title"])

# # Function to search movies based on keywords
# def search(title):
#     title = title.lower().replace('[^a-zA-Z0-9\s]', '')
#     query_vec = vectorizer.transform([title])
#     similarity = cosine_similarity(query_vec, tfidf).flatten()
#     indices = np.argpartition(similarity, -10)[-10:]
#     results = movies.iloc[indices].iloc[::-1]
#     return results

# # Function to get top-N recommendations for a user
# def get_recommendations(user_id, n=10):
#     user_ratings = ratings[ratings['userId'] == user_id]
#     movies_rated = user_ratings['movieId'].tolist()
    
#     # Get the movies not rated by the user
#     unrated_movies = movies[~movies['movieId'].isin(movies_rated)]['movieId']
    
#     # Predict ratings for unrated movies
#     predictions = [algo.predict(user_id, movie_id)[3] for movie_id in unrated_movies]
    
#     # Sort the predictions and get the top-N recommendations
#     top_n = sorted(zip(unrated_movies, predictions), key=lambda x: x[1], reverse=True)[:n]
    
#     # Get the movie titles and genres for the recommendations
#     recommendations = []
#     for movie_id, _ in top_n:
#         movie = movies[movies['movieId'] == movie_id].iloc[0]
#         recommendations.append({
#             'title': movie['title'],
#             'genres': movie['genres']
#         })
    
#     return recommendations

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         user_id = request.form['user_id']
#         session['user_id'] = user_id
#         return render_template('search.html')
#     return render_template('login.html')

# @app.route('/search', methods=['GET', 'POST'])
# def search_route():
#     if 'user_id' not in session:
#         return render_template('login.html')
    
#     if request.method == 'POST':
#         search_term = request.form['search_term']
#         search_results = search(search_term)
#         recommendations = get_recommendations(session['user_id'])
#         return render_template('recommendations.html', recommendations=recommendations, search_results=search_results)
    
#     return render_template('search.html')

# if __name__ == '__main__':
#     app.run(debug=True)





























from flask import Flask, render_template, request, session
import pandas as pd
from surprise import Reader, Dataset, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load the data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Create the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set the template directory
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app.template_folder = template_dir

# Create a reader and load the data into a surprise dataset
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train the recommendation model
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Preprocess movie titles for search
movies['clean_title'] = movies['title'].str.lower().str.replace('[^a-zA-Z0-9\s]', '')
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Initialize a dictionary to store recommendations for each user
user_recommendations = {}

# Function to search movies based on keywords
def search(title):
    title = title.lower().replace('[^a-zA-Z0-9\s]', '')
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = movies.iloc[indices].iloc[::-1]
    return results

# Function to get top-N recommendations for a user
def get_recommendations(user_id, n=10):
    if user_id in user_recommendations:
        return user_recommendations[user_id]

    user_ratings = ratings[ratings['userId'] == user_id]
    movies_rated = user_ratings['movieId'].tolist()

    # Get the movies not rated by the user
    unrated_movies = movies[~movies['movieId'].isin(movies_rated)]['movieId']

    # Predict ratings for unrated movies
    predictions = [algo.predict(user_id, movie_id)[3] for movie_id in unrated_movies]

    # Sort the predictions and get the top-N recommendations
    top_n = sorted(zip(unrated_movies, predictions), key=lambda x: x[1], reverse=True)[:n]

    # Get the movie titles and genres for the recommendations
    recommendations = []
    for movie_id, _ in top_n:
        movie = movies[movies['movieId'] == movie_id].iloc[0]
        recommendations.append({
            'title': movie['title'],
            'genres': movie['genres']
        })

    user_recommendations[user_id] = recommendations
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['user_id']
        session['user_id'] = user_id
        return render_template('search.html')
    return render_template('login.html')

@app.route('/search', methods=['GET', 'POST'])
def search_route():
    if 'user_id' not in session:
        return render_template('login.html')

    if request.method == 'POST':
        search_term = request.form['search_term']
        search_results = search(search_term)
        recommendations = get_recommendations(session['user_id'])
        return render_template('recommendations.html', recommendations=recommendations, search_results=search_results)

    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True)