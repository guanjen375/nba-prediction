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

data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 'OU-Cover', 'OU', 'TEAM_ID', 'TEAM_ID.1'],
          axis=1, inplace=True)
data.drop(['TM_TOV_PCT_RANK','EFG_PCT_RANK','TS_PCT_RANK' ,'PACE_RANK','PIE_RANK',
           'AST_TO_RANK','AST_RATIO_RANK','OREB_PCT_RANK','DREB_PCT_RANK','REB_PCT_RANK',
            'MIN_RANK','OFF_RATING_RANK','DEF_RATING_RANK','NET_RATING_RANK','AST_PCT_RANK',
            'GP_RANK','W_RANK','L_RANK','W_PCT_RANK'],axis=1,inplace=True)
data.drop(['TM_TOV_PCT_RANK.1','EFG_PCT_RANK.1','TS_PCT_RANK.1' ,'PACE_RANK.1','PIE_RANK.1',
           'AST_TO_RANK.1','AST_RATIO_RANK.1','OREB_PCT_RANK.1','DREB_PCT_RANK.1','REB_PCT_RANK.1',
           'MIN_RANK.1','OFF_RATING_RANK.1','DEF_RATING_RANK.1','NET_RATING_RANK.1','AST_PCT_RANK.1',
           'GP_RANK.1','W_RANK.1','L_RANK.1','W_PCT_RANK.1'],axis=1,inplace=True)
print(data)