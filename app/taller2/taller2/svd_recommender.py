import pandas as pd
import pickle
from surprise import SVD
from surprise import Reader
from surprise import Dataset
from collections import defaultdict

from pymongo import MongoClient

df = pd.read_csv( '../notebooks/toronto_reviews.csv')
df2 = df[[ 'user_id', 'business_id', 'stars'] ]
reader = Reader( rating_scale = ( 1, 5 ) )
train_data = Dataset.load_from_df( df2[ [ 'user_id', 'business_id', 'stars' ] ], reader )
train_data = train_data.build_full_trainset()

algo_pkl = open('../notebooks/svd_algo.pickle', 'rb')
algo = pickle.load(algo_pkl)
algo_pkl.close()


def get_recommendations_for_user(user_id):

    def get_top_n(predictions, n = 10):
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key = lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    test = [x for x in train_data.build_testset() if x[0] == user_id]

    predictions = algo.test(test)
    
    top_n = get_top_n(predictions, n = 10)
    top_n = dict(top_n)
    try:
        return top_n[user_id]
    except KeyError:
        print('recommendations not found')
        return {}
