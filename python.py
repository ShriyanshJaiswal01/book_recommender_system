import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

books = pd.read_csv(r'dataset\Books.csv\Books.csv')
ratings = pd.read_csv(r'dataset\Ratings.csv\Ratings.csv')
users = pd.read_csv(r'dataset\Users.csv\Users.csv')

ratings_with_name = ratings.merge(books, on='ISBN')
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)


# print("\n")

ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)
# print(avg_rating_df.head())
# print("\n")

popular_df = avg_rating_df.merge(num_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_ratings', ascending=False).head(50)
# print(popular_df)
 
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author', 'num_ratings', 'avg_ratings','Image-URL-M']]
pickle.dump(popular_df, open('popular.pkl', 'wb'))

x = ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
padhe_likhe_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
# print(y)

famous_books = y[y].index
print(famous_books)

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
# print(final_ratings)

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
# pt_reset = pt.reset_index()

similarity_score = cosine_similarity(pt)
# print(similarity_score[0])

# def recommend_books(book_name):

# print(pt)
# book_name = 'Zoya'
# row_pos = pt_reset[pt_reset['Book-Title'] == book_name].index

# if len(row_pos) > 0:
#     print("Row position is:", row_pos[0])
# else:
#     print(f"'{book_name}' not found in 'Book-Title' column.")

def recommend_books(book_name):
  index = np.where(pt.index == book_name)[0][0]  # Get the index of the book
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

  return data  

recommend_books('Zoya')

pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_score, open('similarity_score.pkl', 'wb'))

# print(np.where(pt.index == 'You Belong To Me')[0][0])  # Get the index of the book 'Zoya'

