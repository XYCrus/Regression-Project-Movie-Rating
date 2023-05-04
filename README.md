# Regression-Project-Movie-Rating

## Preamble:

In 2006, Netflix released over 100 million ratings from over 400,000 users on over 17,000 films. Whoever improved the RMSE of Netflix’s “Cinematch” algorithm by 10% within 5 years received $1,000,000. Here, I revisit this notorious contest by using a subset of this dataset, with a different test set than the original contest.

## Command:

### 1. python txt_2_df.py ../data/data.txt

* *preprocess data.txt into customized csv format under result folder*

### 2. python training.py ../result/input.csv > ../result/output.txt

* *train data with chunks by CatBoostRegressor*