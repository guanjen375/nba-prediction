import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

dataset = "dataset_2012-23"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

data.drop(['TEAM_NAME', 'TEAM_NAME.1','TEAM_NAME','TEAM_NAME.1','TEAM_ID','TEAM_ID.1'],
          axis=1, inplace=True)
data.drop(['GP_RANK','W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK','FG_PCT_RANK', 
           'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK','FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 
           'DREB_RANK', 'REB_RANK','AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 
           'PF_RANK','PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK'],axis=1,inplace=True)
data.drop(['GP_RANK.1', 'W_RANK.1', 'L_RANK.1','W_PCT_RANK.1', 'MIN_RANK.1', 'FGM_RANK.1', 'FGA_RANK.1',
           'FG_PCT_RANK.1', 'FG3M_RANK.1', 'FG3A_RANK.1', 'FG3_PCT_RANK.1',
           'FTM_RANK.1', 'FTA_RANK.1', 'FT_PCT_RANK.1', 'OREB_RANK.1',
           'DREB_RANK.1', 'REB_RANK.1', 'AST_RANK.1', 'TOV_RANK.1', 'STL_RANK.1',
           'BLK_RANK.1', 'BLKA_RANK.1', 'PF_RANK.1', 'PFD_RANK.1', 'PTS_RANK.1','PLUS_MINUS_RANK.1'],axis=1,inplace=True)



print(data)
print(data.columns)