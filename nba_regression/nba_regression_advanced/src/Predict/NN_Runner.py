import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from colorama import Fore, Style, init, deinit
from tensorflow.keras.models import load_model
from src.Utils import Expected_Value

init()
model = load_model('Models/Trained-Model-ML')
ou_model = load_model("Models/Trained-Model-OU")


def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(model.predict(np.array([row]),verbose=0))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)
    data = tf.keras.utils.normalize(data, axis=1)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(ou_model.predict(np.array([row]),verbose=0))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        under_over = int(np.argmax(ou_predictions_array[count]))
        denominator = ou_predictions_array[count][0][0]+ou_predictions_array[count][0][1]
        uo_str = ''
        if(under_over == 0):
            uo_confidence = round(ou_predictions_array[count][0][0]/denominator * 100, 1)
            uo_str = Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + ' (' + str(uo_confidence) + '%)' + Style.RESET_ALL
        else:
            uo_confidence = round(ou_predictions_array[count][0][1]/denominator * 100, 1)
            uo_str = Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + ' (' + str(uo_confidence) + '%)' + Style.RESET_ALL
        ev_home = float(Expected_Value.expected_odd(ml_predictions_array[count][0][1]))
        ev_away = float(Expected_Value.expected_odd(ml_predictions_array[count][0][0]))
        if ev_home > ev_away:
            print(home_team + Fore.GREEN + ' (' + str(ev_home) + ')' + Style.RESET_ALL + ' @ ' + away_team + Fore.RED + ' (' + str(ev_away) + ')' + Style.RESET_ALL + ' : ' + uo_str)
        else:
            print(home_team + Fore.RED + ' (' + str(ev_home) + ')' + Style.RESET_ALL + ' @ ' + away_team + Fore.GREEN + ' (' + str(ev_away) + ')' + Style.RESET_ALL + ' : ' + uo_str)

        count += 1


    deinit()

