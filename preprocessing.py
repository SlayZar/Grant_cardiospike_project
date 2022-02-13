import pandas as pd
import numpy as np

# Генерация фичей
def features(df):
    for i in range(1,8):
        df[f'x-{i}'] = df.groupby(['id'])['x'].diff(-i)
        df[f'x+{i}'] = df.groupby(['id'])['x'].diff(i)
    
    for i in range(2,8):
        df[f'mult_{i}'] = df[f'x-{i-1}'] * df[f'x-{i}']
        df[f'mult_pl_{i}'] = df[f'x+{i-1}'] * df[f'x+{i}']
        
    df = df.merge(df.groupby(['id'])['x'].median().to_frame('median_x'), left_on = ['id'], right_index=True)
    df = df.merge(df.groupby(['id'])['x'].rolling(6, min_periods = 2, center=True).max().to_frame('max_6'),
             on = np.arange(len(df))).drop(['key_0'], axis=1)
    df = df.merge(df.groupby(['id'])['x'].rolling(6, min_periods = 2, center=True).min().to_frame('min_6'),
             on = np.arange(len(df))).drop(['key_0'], axis=1)
    df = df.merge(df.groupby(['id'])['x'].rolling(12, min_periods = 2, center=True).std().to_frame('std12_c'),
             on = np.arange(len(df))).drop(['key_0'], axis=1)
    df = df.merge(df.groupby(['id'])['x'].rolling(12, min_periods = 2).std().to_frame('std12'),
             on = np.arange(len(df))).drop(['key_0'], axis=1)
    df = df.merge(df.groupby(['id'])['x'].rolling(6, min_periods = 2, center=True).std().to_frame('std6_c'),
             on = np.arange(len(df))).drop(['key_0'], axis=1)
    df = df.merge(df.groupby(['id'])['x'].rolling(6, min_periods = 2).std().to_frame('std6'),
             on = np.arange(len(df))).drop(['key_0'], axis=1)
    df['min_6_norm'] = df['min_6'] / df['median_x']
    df['max_6_norm'] = df['max_6'] / df['median_x']
    df['med_share'] = (df['max_6'] - df['median_x']) / (df['median_x'] - df['min_6'])
    
    df = df.merge(df.groupby(['id'])['x-1'].rolling(10, min_periods = 2).max().to_frame('max10'),
             on = np.arange(len(df))).drop(['key_0'], axis=1)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=10)
    df = df.merge(df.groupby(['id'])['x-1'].rolling(window=indexer, min_periods = 2).max().to_frame('max10_reverse'),
                 on = np.arange(len(df))).drop(['key_0'], axis=1)
    df['10_same'] = (df['max10'] == df['max10'].shift(-1)).astype(int)
    df['10_same_reverse'] = (df['max10_reverse'] == df['max10_reverse'].shift(-1)).astype(int)
    for i in range(1,10):
        df['10_same_shift'] = df['10_same'].shift(-i)
        df[f'my_feat{i}'] = df['10_same'] + df['10_same_shift'] 
    for i in range(1,10):
        df['10_same_shift_res'] = df['10_same_reverse'].shift(-i)
        df[f'my_feat_2_{i}'] = df['10_same_reverse'] + df['10_same_shift_res'] 
    df['same_feat'] = df[['my_feat1','my_feat2', 'my_feat3', 'my_feat4', 'my_feat5', 'my_feat6', 'my_feat7', 
        'my_feat8', 'my_feat9']].idxmin(axis=1).fillna('my_feat0').apply(lambda x: int(str(x).split('feat')[-1]))
    df['same_feat2'] = df[['my_feat_2_1','my_feat_2_2', 'my_feat_2_3', 'my_feat_2_4', 'my_feat_2_5', 'my_feat_2_6', 
                           'my_feat_2_7', 'my_feat_2_8', 'my_feat_2_9']].idxmin(axis=1)\
                                .fillna('my_feat_2_0').apply(lambda x: int(str(x).split('feat_2_')[-1]))
    df['same_feat_3'] = df.groupby(['id'])['same_feat'].shift(-3)
    df['same_feat_6'] = df.groupby(['id'])['same_feat'].shift(-6)
    df['same_feat_10'] = df.groupby(['id'])['same_feat'].shift(-10)
    drops = ['10_same_reverse', '10_same_shift_res', '10_same_shift']
    for i in df.columns:
        if 'my' in i:
            drops.append(i)
    df.drop(drops, axis=1, inplace=True)
    return df