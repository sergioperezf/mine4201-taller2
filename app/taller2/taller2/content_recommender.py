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


def get_user_top_reviews(user_id):
    
    review_indexes = dfReviews.loc[dfReviews['user_id_x'] == 'zsZVg16yjZu5NIiS0ayjrQ'].sort_values('stars_x', ascending=False).head(10).index

    businesess_count = {}
    businesses_top = {}
    for index in review_indexes:
        cosine_similarities = cosine_similarity(tfidf_matrix[index], tfidf_matrix)
        for i in cosine_similarities[0].argsort()[:-10:-1]:
            business_id = dfReviews.loc[i]['business_id_x']
            if businesess_count.get(business_id):
                businesess_count[business_id] = businesess_count[business_id] + 1
                businesses_top[business_id] = max(cosine_similarities[0][i], businesses_top[business_id])
            else:
                businesess_count[business_id] = 1
                businesses_top[business_id] = cosine_similarities[0][i]
            #print(list(db.business.find({'business_id': dfReviews.loc[i]['business_id_x']}))[0]['name'], cosine_similarities[0][i])
            
    top_businesses_count = {}
    max_b = 10
    i = 0
    for business in sorted(businesess_count, key=businesess_count.get, reverse=True):
        if i >= max_b:
            break
        if businesses_top[business] < 1:
            top_businesses_count[business] = businesses_top[business]
            i = i + 1

    for business in sorted(top_businesses_count, key=top_businesses_count.get, reverse=True):
        print(list(db.business.find({'business_id': business}))[0]['name'])
get_user_top_reviews('zsZVg16yjZu5NIiS0ayjrQ')