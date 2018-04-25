from django.shortcuts import render
from . import recommender
from pymongo import MongoClient

client = MongoClient()
db = client['test']

def user(request):
    user_id = (request.GET['user'])
    user = db.user.find({'user_id': user_id})[0]
    content, svd, collab = recommender.get_full_recommendation(user_id)
    content_human = {}
    svd_human = {}
    collab_human = {}
    for rec in content:
        content_human[db.business.find({'business_id': rec[0]})[0]['name']] = rec[1]
    for rec in svd:
        svd_human[db.business.find({'business_id': rec[0]})[0]['name']] = rec[1]
    for rec in collab:
        collab_human[db.business.find({'business_id': rec[0]})[0]['name']] = rec[1]
    return render(request, 'user.html', {
        'user': user,
        'content': content_human,
        'svd': svd_human,
        'collab': collab_human
        })

def index(request):
    return render(request, 'index.html')


