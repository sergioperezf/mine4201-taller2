from django.shortcuts import render
from . import recommender
from pymongo import MongoClient

client = MongoClient()
db = client['test']

def user(request):
    user_id = (request.GET['user'])
    user = db.user.find({'user_id': user_id})[0]
    recommendations = recommender.get_full_recommendation(user_id)
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    recommendations_human = {}
    for rec in recommendations:
        recommendations_human[db.business.find({'business_id': rec[0]})[0]['name']] = rec[1]
        
    return render(request, 'user.html', {
        'user': user,
        'recommendations': recommendations_human
        })

def index(request):
    return render(request, 'index.html')


