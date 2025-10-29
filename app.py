

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movie data and similarity matrix
movie_matrix = pd.read_csv('movie_matrix.csv', index_col=0)
cosine_sim = cosine_similarity(movie_matrix.T)
similarity_df = pd.DataFrame(cosine_sim, index=movie_matrix.columns, columns=movie_matrix.columns)

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    data = request.get_json()
    movie = data['movie']
    if movie not in similarity_df.columns:
        return jsonify({'error': 'Movie not found!'})
    similar_movies = similarity_df[movie].sort_values(ascending=False)[1:6]
    return jsonify({'recommended_movies': list(similar_movies.index)})

if __name__ == '__main__':
    app.run(debug=True)

