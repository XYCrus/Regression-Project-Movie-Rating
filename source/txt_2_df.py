#%%
import os
import sys
import time
import pandas as pd

#%%
def process_chunk(chunk, current_movie):
    data = []
    for line in chunk:
        line = line.strip()

        if line.endswith(':'):
            current_movie = line[:-1]

        else:
            row_data = line.split(',')
            row_data.insert(0, current_movie)
            data.append(row_data)
    
    df = pd.DataFrame(data, columns = ['Movie', 
                                       'ID', 
                                       'Rating', 
                                       'Date'])
    
    df['Movie'] = df['Movie'].astype('category')
    df['ID'] = df['ID'].astype('category')
    df['Rating'] = df['Rating'].astype(int)
    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    df.drop('Date', axis = 1, inplace = True)
    
    return df, current_movie

def txt_2_df(file_path, chunksize = 10000):
    dataframes = []
    current_movie = None
    chunk = []

    with open(file_path, 'r') as file:
        for line in file:
            if len(chunk) == chunksize:
                df, current_movie = process_chunk(chunk, current_movie)
                dataframes.append(df)
                chunk = []

            chunk.append(line)
    
    if chunk:
        df, current_movie = process_chunk(chunk, current_movie)
        dataframes.append(df)
    
    return pd.concat(dataframes)

def df_to_file(df, name):
    if not os.path.exists('../result'):
        os.mkdir('../result')

    df.to_csv("../result/{}.csv".format(name))

#%%
if __name__ == '__main__':
    start_time = time.time()

    filepath = sys.argv[1]

    df = txt_2_df(filepath)

    df_to_file(df, 'input')

    duration = time.time() - start_time
    print("%.0f seconds in total" % duration)