import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


rating_table = pickle.load(open("rating_table.pkl","rb"))
books_image_data = pickle.load(open("books_image_data.pkl","rb"))

books_name = rating_table.index.to_list()


sparse_matrix = csr_matrix(rating_table)
model = NearestNeighbors(algorithm='brute')
model.fit(sparse_matrix)


#Function for recommending books
def recommend(book_name):
  recommended_books = []
  image_url = []
  book_index = np.where(rating_table.index==book_name)[0][0]
  distances , suggestions = model.kneighbors(rating_table.iloc[book_index,:].values.reshape(1,-1),n_neighbors=5)
  suggestions = np.ravel(suggestions, order='C') #2d to 1d array
  for i in suggestions:
    recommended_books.append(rating_table.index[i])

  for i in recommended_books:
    image_url.append(books_image_data[books_image_data["title"] == i ].image.to_string(index=False))
    
  return recommended_books,image_url







import streamlit as st

st.title("Book Recommendation Engine")

selected_book = st.selectbox(
     'Search your books here',
     books_name)

if st.button('Search'):
    books,images = recommend(selected_book) 

    container1 =st.container()
    container1.header("YOU HAVE SEARCHED FOR -")
    container1.header(books[0])
    container1.image(images[0])

    st.header("PEOPLE ALSO LIKED -")
    col1, col2, col3,col4 = st.columns(4)

    with col1:
        st.subheader(books[1])
        st.image(images[1])
    with col2:
        st.subheader(books[2])
        st.image(images[2])
    with col3:
        st.subheader(books[3])
        st.image(images[3])
    with col4:
        st.subheader(books[4])
        st.image(images[4])




