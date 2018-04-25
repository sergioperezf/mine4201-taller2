import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pymongo import MongoClient

dfReviews = pd.read_pickle('../notebooks/reviews_with_text.pickle')
f = open("../notebooks/tfidf_matrix.pickle","rb")
tfidf_matrix = pickle.load(f)
f.close()


def get_recommendations_for_user(user_id):
    
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
            
    top_businesses_count = {}
    max_b = 10
    i = 0
    for business in sorted(businesess_count, key=businesess_count.get, reverse=True):
        if i >= max_b:
            break
        if businesses_top[business] < 1:
            top_businesses_count[business] = businesses_top[business]
            i = i + 1

    
    return list([(business, top_businesses_count[business]) for business in sorted(top_businesses_count, key=top_businesses_count.get, reverse=True)])
