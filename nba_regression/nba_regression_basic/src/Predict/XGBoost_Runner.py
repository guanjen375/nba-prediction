import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_ML_2012s.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_UO_2012s.json')


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    diff = []

    for row in data:
        diff.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    #frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    score = []

    for row in data:
        score.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))
    
    count = 0
    for game in games:
        sum = score[count][0]
        home_team = game[0]
        away_team = game[1]
        ev_home = diff[count][0]
        ev_away = -diff[count][0]
        score_home = round((float(ev_home)/2+float(sum)/2),1)
        score_away = round((float(ev_away)/2+float(sum)/2),1)
        if ev_home > ev_away:
            print(home_team + Fore.GREEN + ' (' + str(score_home) + ')' + Style.RESET_ALL + ' @ ' + away_team + Fore.RED + ' (' + str(score_away) + ')' + Style.RESET_ALL)
        else:
            print(home_team + Fore.RED + ' (' + str(score_home) + ')' + Style.RESET_ALL + ' @ ' + away_team + Fore.GREEN + ' (' + str(score_away) + ')' + Style.RESET_ALL)

        count += 1


    deinit()
