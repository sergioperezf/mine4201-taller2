from . import content_recommender, svd_recommender, user_user_recommender
import pandas as pd


tabla = pd.read_csv('../notebooks/users_by_review.csv', sep = '\t')

def get_percentage_collab(user_id):
    total = tabla.shape[0]
    index = tabla.loc[tabla['id'] == user_id].index[0] + 1
    return (index/total * .6) + .2


def get_weights(user_id):
    collab = get_percentage_collab(user_id)
    other = (1.0 - collab) / 2.0
    return collab, other, other


def get_full_recommendation(user_id):
    content_recommendations = content_recommender.get_recommendations_for_user(user_id)
    svd_recommendations = svd_recommender.get_recommendations_for_user(user_id)
    user_user_recommendations = user_user_recommender.get_recommendations_for_user(user_id)

    # get weights
    collab_weight, svd_weight, content_weight = get_weights(user_id)

    content_recommendations_calculated = [(rec[0], rec[1] * content_weight) for rec in content_recommendations]
    svd_recommendations_calculated = [(rec[0], rec[1] * svd_weight) for rec in svd_recommendations]
    user_user_recommendations_calculated = [(rec[0], rec[1] * collab_weight) for rec in user_user_recommendations]
    
    return content_recommendations_calculated + svd_recommendations_calculated + user_user_recommendations_calculated