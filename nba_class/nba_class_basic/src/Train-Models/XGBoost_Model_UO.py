import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import optuna


dataset = "dataset_2012-23"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()
OU = data['OU-Cover']
total = data['OU']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU','TEAM_ID','TEAM_ID.1'],
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

data['OU'] = np.asarray(total)
print(data.columns)
data = data.values
data = data.astype(float)
max_value = 0
epochs=250
params = []

def train_parameter(trial):
    x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.1)
    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)
    evals = [(train,'train'),(test,'val')]
    params = {
        'objective': 'binary:logistic',
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'random_state': trial.suggest_int('random_state', 1, 1000)
    }
    model = xgb.train(params, train, epochs, evals = evals, verbose_eval=False)
    prediction = model.predict(xgb.DMatrix(x_test))
    y_pred = []
    for z in prediction:
        if(z>0.95 or z<0.05):
            return 0
        elif(z>0.5):
            y_pred.append(1)
        else:
            y_pred.append(0)
    return accuracy_score(y_test,y_pred)

for x in range(100):
    study = optuna.create_study(direction='maximize',pruner=optuna.pruners.HyperbandPruner())
    optuna.logging.set_verbosity(optuna.logging.WARN)
    study.optimize(train_parameter, n_trials = 250,n_jobs=-1)
    print('Best score:', study.best_value)

    params = study.best_trial.params
    params['objective'] = 'binary:logistic'

    for y in range(200):

        x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.1)
        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test, label=y_test)
        evals = [(train,'train'),(test,'val')]

        model = xgb.train(params, train, epochs, evals = evals, verbose_eval=False)
        prediction = model.predict(xgb.DMatrix(x_test))
        acc = 0
        y_pred = []
        for z in prediction:
            if(z>0.95 or z<0.05):
                pass
            elif(z>0.5):
                y_pred.append(1)
            else:
                y_pred.append(0)
        if(len(y_pred) == len(y_test)):
            acc = accuracy_score(y_test,y_pred)
        if(acc > max_value):
            max_value = acc
            print('Save Model:XGBoost_{}_UO_2012s.json'.format(acc))
            model.save_model('../../Models/XGBoost_{}_UO_2012s.json'.format(acc))
