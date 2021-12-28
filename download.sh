#!/bin/bash

wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz
gunzip reviews_Home_and_Kitchen_5.json.gz
python preprocessing.py home_and_kitchen

wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-100k.zip 
wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-1m.zip
wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-10m.zip
wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-20m.zip
wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-100k.zip
unzip ml-1m.zip
unzip ml-10m.zip
unzip ml-20m.zip
unzip ml-25m.zip
python preprocessing.py 100k
python preprocessing.py 1m
python preprocessing.py 10m
python preprocessing.py 20m
python preprocessing.py 25m
