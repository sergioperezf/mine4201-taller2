import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pymongo import MongoClient

client = MongoClient()
db = client['test']
dfReviews = pd.read_pickle('../notebooks/reviews_with_text.pickle')

f = open("../notebooks/tfidf_matrix.pickle","rb")
tfidf_matrix = pickle.load(f)

def item(id):
    return dfReviews.loc[dfReviews['review_id'] == id]

def get_recommendation_by_review(review_id):
    index = item(review_id).index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[index], tfidf_matrix)
    for i in cosine_similarities[0].argsort()[:-10:-1]:
        print(list(db.business.find({'business_id': dfReviews.loc[i]['business_id_x']}))[0]['name'])


get_recommendation_by_review('---0hl58W-sjVTKi5LghGw')