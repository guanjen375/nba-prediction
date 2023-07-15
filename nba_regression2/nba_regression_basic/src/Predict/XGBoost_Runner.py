import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_home = xgb.Booster()
xgb_home.load_model('Models/XGBoost_Home.json')
xgb_away = xgb.Booster()
xgb_away.load_model('Models/XGBoost_Away.json')


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    point_h = []
    point_a = []

    for row in data:
        point_h.append(xgb_home.predict(xgb.DMatrix(np.array([row]))))
        point_a.append(xgb_away.predict(xgb.DMatrix(np.array([row]))))

    count = 0
    for game in games:
        p_h = round(point_h[count][0],1)
        p_a = round(point_a[count][0],1)
        home_team = game[0]
        away_team = game[1]
        if p_h > p_a:
            print(home_team + Fore.GREEN + ' (' + str(p_h) + ')' + Style.RESET_ALL + ' @ ' + away_team + Fore.RED + ' (' + str(p_a) + ')' + Style.RESET_ALL)
        else:
            print(home_team + Fore.RED + ' (' + str(p_h) + ')' + Style.RESET_ALL + ' @ ' + away_team + Fore.GREEN + ' (' + str(p_a) + ')' + Style.RESET_ALL)
        count += 1



    deinit()
