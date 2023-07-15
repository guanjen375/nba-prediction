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
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))
    
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        uo_str = ""
        under_over = round(ou_predictions_array[count][0]*100,1)
        if(under_over > 50):
            uo_str = Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + ' (' + str(under_over) + '%)' + Style.RESET_ALL
        else:
            uo_str = Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + ' (' + str(100-under_over) + '%)' + Style.RESET_ALL
        ev_home = float(Expected_Value.expected_odd(ml_predictions_array[count][0]))
        ev_away = float(Expected_Value.expected_odd(1-ml_predictions_array[count][0]))
        if ev_home > ev_away:
            print(home_team + Fore.GREEN + ' (' + str(ev_home) + ')' + Style.RESET_ALL + ' @ ' + away_team + Fore.RED + ' (' + str(ev_away) + ')' + Style.RESET_ALL + ' : ' + uo_str)
        else:
            print(home_team + Fore.RED + ' (' + str(ev_home) + ')' + Style.RESET_ALL + ' @ ' + away_team + Fore.GREEN + ' (' + str(ev_away) + ')' + Style.RESET_ALL + ' : ' + uo_str)

        count += 1


    deinit()
