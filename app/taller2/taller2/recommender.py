from . import content_recommender, svd_recommender, user_user_recommender

def get_full_recommendation(user_id):
    return content_recommender.get_recommendations_for_user(user_id), svd_recommender.get_recommendations_for_user(user_id), user_user_recommender.get_recommendations_for_user(user_id)