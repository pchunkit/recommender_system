import streamlit as st
import numpy as np 
import pandas as pd 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

rating = pd.io.parsers.read_csv('ml-1m/ratings.dat', 
                                names=['user_id', 'movie_id', 'rating', 'time'],
                                engine='python', delimiter='::')
movies = pd.io.parsers.read_csv('ml-1m/movies.dat',
                                names=['movie_id', 'title', 'genre'],
                                engine='python', delimiter='::')
users = pd.io.parsers.read_csv('ml-1m/users.dat',
                                names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                                engine='python', delimiter='::')


# from wordcloud import WordCloud
# import matplotlib.pyplot as plt 
# def plot_wordcloud(): 
#     wordcloud = WordCloud(width = 500, height = 500,  background_color ='black').generate(" ".join([x for x in movies['genre']]))
#     fig, ax = plt.subplots()
#     ax.imshow(wordcloud)
#     ax.axis("off")
#     fig.tight_layout(pad=-1)
#     return st.pyplot(fig)
    
class content_based_filtering:
    def __init__(self): 
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        # compaare similarity according to movie genre
        tfidf_matrix = tf.fit_transform(movies['genre']) 
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    def calc(self, movie_title): 
        indices = list(enumerate(self.similarity_matrix[movies[movies['title'] == movie_title].index][0]))
        sort_scores = sorted(indices, key=lambda x: x[1], reverse=True)
        top_10 = sort_scores[1:11] 
        top_10_idx = [x[0] for x in top_10]
        recommended_movies = movies.loc[top_10_idx, 'title'].tolist()
        return recommended_movies, [x[1] for x in top_10]
    
class collaborative_based_filtering: 
    def __init__(self):
        if os.path.exists("svd_matrix.npy"): 
            self.svd_matrix = np.load("svd_matrix.npy")
        else: 
            ratings_mat = np.ndarray(shape=(np.max(rating.movie_id.values), np.max(rating.user_id.values)),dtype=np.uint8)
            ratings_mat[rating.movie_id.values-1, rating.user_id.values-1] = rating.rating.values
            normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
            A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
            U, sigma, V = np.linalg.svd(A)
            self.svd_matrix = V 
            np.save("svd_matrix", self.svd_matrix)
            
    def svd_cosine_similarity(self, data, movie_id, top_n=10):
        index = movie_id - 1 # Movie id starts from 1
        movie_row = data[index, :]
        magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
        similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
        sort_indexes = np.argsort(-similarity)
        top_indexes = sort_indexes[:top_n]
        top_similarity = [similarity[index] for index in top_indexes]
        recommended_movies = [movies.loc[movies["movie_id"]==(i + 1) , "title"].values[0] for i in top_indexes]
        return recommended_movies, [(i+1) for i in top_indexes], top_similarity
            
    def calc(self, movie_title): 
        top_n = 20 
        k = 50 
        sliced = self.svd_matrix.T[:, :k] 
        movie_id = movies.loc[movies["title"]==movie_title, "movie_id"].values[0]
        svd_recommended_movies = self.svd_cosine_similarity(sliced, movie_id, top_n)
        return svd_recommended_movies
        
    

if __name__ == "__main__": 
#     st.set_page_config(layout="wide")
#     content_recommender = content_based_filtering()
    collab_recommender = collaborative_based_filtering()
    # streamlit apps
    st.title('Movie Recommender System')
    
    movie_option = st.selectbox("Select a movie:", 
                                list(movies['title'].unique()))
    st.write("Selected movie: ", movie_option)
    st.write("Genre of {}: {}".format(movie_option, movies.loc[movies['title']==movie_option, 'genre'].item()))
#     recommended_ls = [] 
    
#     st.write("Collaborative-based Filtering")
    recommended_movies, top_index, similarity_score = collab_recommender.calc(movie_option)
#     st.write(recommended_movies)
#     recommended_ls.extend(recommended_movies)
#     st.write("Content-based Filtering")
#     recommended_movies, similarity_score = content_recommender.calc(movie_option)
#     st.write(recommended_movies)
#     recommended_ls.extend(recommended_movies)
#     recommended_ls = list(set(recommended_ls))
    ncol = 4 
    for i in range(int(len(recommended_movies)/ncol)):
        cols = st.columns(ncol)
        for idx, (m_name, m_id) in enumerate(zip(recommended_movies[i*ncol: (i*ncol)+4], top_index[i*ncol: (i*ncol)+4])):
            genre = movies.loc[movies['movie_id']==m_id, 'genre'].values[0].split("|")
            avg_rating = np.mean(rating[rating["movie_id"] == m_id]['rating'])
            movie_info = "Genre: \n{} \n\nAverage Rating: {:.1f}".format("\n".join([g for g in genre]), avg_rating)
            
            cols[idx].subheader(m_name)
            cols[idx].markdown(movie_info)
            