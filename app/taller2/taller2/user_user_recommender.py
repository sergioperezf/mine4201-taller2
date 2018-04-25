import pickle
from surprise.model_selection import cross_validate, train_test_split, KFold
from surprise import Dataset, Reader, SVD, NormalPredictor, accuracy, evaluate
from collections import defaultdict
from surprise.prediction_algorithms.knns import *

from pymongo import MongoClient

dataset = pickle.load(open('../notebooks/toronto_reviews_array.pickle', 'rb'))
surprise_ds = dataset.stack()
surprise_ds.reset_index(level = [0, 1], inplace = True)
surprise_ds = surprise_ds[surprise_ds.stars != 0]
kf = KFold(n_splits = 2)
reader = Reader(rating_scale = (1, 5))

data = Dataset.load_from_df(surprise_ds[['user', 'business', 'stars']], reader)

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

    train_data = data.build_full_trainset()
    sim_options = {'name': 'msd', 'user_based': False, 'min_k': 10, 'k': 15}
    algo = KNNBasic(sim_options = sim_options)
    algo.fit(train_data)

    test = [x for x in train_data.build_testset() if x[0] == user_id]

    predictions = algo.test(test)
    
    top_n = get_top_n(predictions, n = 10)
    top_n = dict(top_n)
    try:
        return top_n[user_id]
    except KeyError:
        print('recommendations not found')
        return {}
