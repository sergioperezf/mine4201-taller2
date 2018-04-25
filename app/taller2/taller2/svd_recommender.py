import pandas as pd
import pickle
from surprise import SVD
from surprise import Reader
from surprise import Dataset

df = pd.read_csv( '../notebooks/toronto_reviews.csv')
df2 = df[[ 'user_id', 'business_id', 'stars'] ]
reader = Reader( rating_scale = ( 1, 5 ) )
train_data = Dataset.load_from_df( df2[ [ 'user_id', 'business_id', 'stars' ] ], reader )
train_data = train_data.build_full_trainset()
mean = train_data.global_mean

algo_pkl = open('../notebooks/svd_algo.pickle', 'rb')
algo = pickle.load(algo_pkl)
algo_pkl.close()