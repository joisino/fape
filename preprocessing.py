import sys
import json

import numpy as np
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise import Reader


dataset = sys.argv[1]
if dataset == 'home_and_kitchen':
    sr = {}
    si = {}
    n = 66519
    m = 28237
    R = np.zeros((n, m))
    with open('./reviews_Home_and_Kitchen_5.json') as f:
        for r in f:
            s = json.loads(r)
            if s['reviewerID'] not in sr:
                sr[s['reviewerID']] = len(sr)
            if s['asin'] not in si:
                si[s['asin']] = len(si)
            R[sr[s['reviewerID']], si[s['asin']]] = s['overall']

if dataset == 'home_and_kitchen':
    K = 10
    while(1):
        prv = R.copy()
        mask = R > 0
        R[:, mask.sum(0) < K] = 0
        mask = R > 0
        R[mask.sum(1) < K, :] = 0
        if (R == prv).all():
            break
    cR = R[R.sum(1) > 0]
    cR = cR[:, cR.sum(0) > 0]
    R = cR
    n, m = R.shape
else:
    if dataset == '100k':
        n = 943
        m = 1682
        filename = 'ml-100k/u.data'
        delimiter = '\t'
    elif dataset == '1m':
        n = 6040
        m = 3706
        filename = 'ml-1m/ratings.dat'
        delimiter = '::'
    elif dataset == '10m':
        n = 69878
        m = 10677
        filename = 'ml-10M100K/ratings.dat'
        delimiter = '::'
    elif dataset == '20m':
        n = 138493
        m = 26744
        filename = 'ml-20m/ratings.csv'
        delimiter = ','
    elif dataset == '25m':
        n = 162541
        m = 59047
        filename = 'ml-25m/ratings.csv'
        delimiter = ','

    R = np.zeros((n, m))

    user_map = {}
    movie_map = {}

    with open(filename) as f:
        if dataset == '20m' or dataset == '25m':
            f.readline()
        for r in f:
            user, movie, r, t = r.split(delimiter)
            if user not in user_map:
                user_map[user] = len(user_map)
            if movie not in movie_map:
                movie_map[movie] = len(movie_map)
            user = user_map[user]
            movie = movie_map[movie]
            r = float(r)
            R[user, movie] = r

    print(n, m)
    print(len(user_map), len(movie_map))

    assert(len(user_map) == n)
    assert(len(movie_map) == m)

mask = R > 0

user_list = []
item_list = []
rating_list = []
for user in range(n):
    for item in range(m):
        if mask[user, item]:
            user_list.append(user)
            item_list.append(item)
            rating_list.append(R[user, item])
df = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'rating': rating_list})
clip = True
if dataset == 'home_and_kitchen' or dataset == '100k' or dataset == '1m':
    reader = Reader(rating_scale=(1, 5))
elif dataset == '10m' or dataset == '20m' or dataset == '25m':
    reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(df, reader)
trainset = data.build_full_trainset()
algo = SVD(random_state=0)
algo.fit(trainset)

for i in range(n):
    for j in range(m):
        if not mask[i, j]:
            R[i, j] = algo.predict(i, j, clip=clip).est

np.save(f'{dataset}.npy', R)
