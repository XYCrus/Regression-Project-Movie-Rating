import sys
import time
import random
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

random.seed(16542515)

def csv_2_df(filepath, chunksize = 100000):
    data_chunks = []
    for chunk in pd.read_csv(filepath, delimiter = ",", dtype = str, chunksize = chunksize):
        data_chunks.append(chunk)

    df = pd.concat(data_chunks, axis = 0, ignore_index = True)
    df.drop(columns = df.columns[0], axis = 1, inplace = True)

    df["Movie"] = df["Movie"].astype("category")
    df["ID"] = df["ID"].astype("category")
    df["Rating"] = df["Rating"].astype(float)
    df["Year"] = df["Year"].astype("category")
    df["Month"] = df["Month"].astype("category")
    df["Day"] = df["Day"].astype("category")

    return df

def random_index(group):
    return random.choice(group.index)

def train_test_split(df):
    grouped = df.groupby('Movie')
    test_indices = grouped.apply(random_index)
    test_df = df.loc[test_indices]

    train_df = df.drop(test_indices)
    train_df.reset_index(drop=True, inplace=True)
    train_df = train_df.sample(frac = 1)
    test_df.reset_index(drop=True, inplace=True)

    xtrain = train_df.drop('Rating', axis=1)
    ytrain = train_df.iloc[:, 2]

    xtest = test_df.drop('Rating', axis=1)
    ytest = test_df.iloc[:, 2]

    return xtrain, ytrain, xtest, ytest

def process_chunk(model, xtrain, ytrain, xtest, ytest, cat_features_indices, first_chunk):
    if len(xtrain) > 0 and len(ytrain) > 0:
        # Incrementally train the model
        model.fit(xtrain, 
                  ytrain, 
                  eval_set = (xtest, ytest), 
                  verbose = 250, 
                  init_model = None if first_chunk else model)
    return model

def DefRegressor(cat_features_indices, iterations = 1000, learning_rate = 0.05, depth = 5):
    model = CatBoostRegressor(
        iterations = iterations,
        learning_rate = learning_rate,
        depth = depth,
        cat_features = cat_features_indices,
        loss_function = 'RMSE',
        verbose = 250
    )

    return model

def run_training(df, chunksize = 3000000):
    xtrain, ytrain, xtest, ytest = train_test_split(df)

    cat_features_indices = list(range(df.drop('Rating', axis = 1).shape[1]))
    model = DefRegressor(cat_features_indices)

    first_chunk = True
    for start_idx in range(0, len(xtrain), chunksize):
        end_idx = min(start_idx + chunksize, len(xtrain))
        model = process_chunk(
            model,
            xtrain[start_idx:end_idx],
            ytrain[start_idx:end_idx],
            xtest,
            ytest,
            cat_features_indices,
            first_chunk
        )
        first_chunk = False

    predictions = model.predict(xtest)

    rmse = mean_squared_error(ytest, predictions, squared = False)

    return rmse

#%%
if __name__ == '__main__':
    start_time = time.time()

    filepath = sys.argv[1]

    df = csv_2_df(filepath)

    rmse = run_training(df)
    print(f'Root Mean Squared Error: {rmse}')

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    