from flask import Flask,render_template,request
import pickle
import requests
import numpy as np

popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_score = pickle.load(open('similarity_score.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           ratings=list(popular_df['avg_ratings'].values),
                           votes=list(popular_df['num_ratings'].values)
                           ) 

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]  # Get the index of the book
    distances = similarity_score[index]  # Get the similarity scores for that book
    similar_items = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
  
    data = []
    for i in similar_items:
     item = []
     temp_df = books[books['Book-Title'] == pt.index[i[0]]]
     temp_df = temp_df.drop_duplicates('Book-Title')
     item.extend(list(temp_df['Book-Title'].values))
     item.extend(list(temp_df['Book-Author'].values))
     item.extend(list(temp_df['Image-URL-M'].values))

     data.append(item)
    
    print(data)
    return render_template('recommend.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)