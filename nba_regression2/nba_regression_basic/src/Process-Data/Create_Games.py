import os
import sqlite3
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Utils.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14, team_index_current

season_array = [ "2012-13", "2013-14", "2014-15", "2015-16",
                 "2016-17", "2017-18", "2018-19","2019-20", "2020-21","2021-22","2022-23"]

df = pd.DataFrame

games = []
days_rest_away = []
days_rest_home = []
score_home = []
score_away = []
teams_con = sqlite3.connect("../../Data/teams_base.sqlite")
odds_con = sqlite3.connect("../../Data/odds.sqlite")

for season in tqdm(season_array):
    odds_df = pd.read_sql_query(f"select * from \"odds_{season}\"", odds_con, index_col="index")
    team_table_str = "teams_{}-{}-" + season
    year_count = 0
    
    for row in odds_df.itertuples():

        home_team = row[3]
        away_team = row[4]

        date = row[2]
        date_array = date.split('-')
        if not date_array or len(date_array) < 2:
            continue
        year = date_array[0] + '-' + date_array[1]
        month = date_array[2][:2]
        day = date_array[2][2:]

        if month[0] == '0':
            month = month[1:]
        if day[0] == '0':
            day = day[1:]
        if int(month) == 1:
            year_count = 1
        end_year_pointer = int(date_array[0]) + year_count
        if end_year_pointer == datetime.now().year:
            if int(month) == datetime.now().month and int(day) >= datetime.now().day:
                continue
            if int(month) > datetime.now().month:
                continue
        
        if(month == '3' and day== '1'):
            continue
        if(year == '2020-21'):
            if(int(month)>=5):
                continue
        else:            
            if(int(month)>=4 and int(month)<=10):
                continue

        team_df = pd.read_sql_query(f"select * from \"teams_{year}-{month}-{day}\"", teams_con, index_col="index")
        if len(team_df.index) == 30:
            days_rest_home.append(row[11])
            days_rest_away.append(row[12])

            sum = row[9]
            win_margin = row[10]
            score_home.append((sum+win_margin)/2)
            score_away.append((sum-win_margin)/2)

            if season == '2007-08':
                home_team_series = team_df.iloc[team_index_07.get(home_team)]
                away_team_series = team_df.iloc[team_index_07.get(away_team)]
            elif season == '2008-09' or season == "2009-10" or season == "2010-11" or season == "2011-12":
                home_team_series = team_df.iloc[team_index_08.get(home_team)]
                away_team_series = team_df.iloc[team_index_08.get(away_team)]
            elif season == "2012-13":
                home_team_series = team_df.iloc[team_index_12.get(home_team)]
                away_team_series = team_df.iloc[team_index_12.get(away_team)]
            elif season == '2013-14':
                home_team_series = team_df.iloc[team_index_13.get(home_team)]
                away_team_series = team_df.iloc[team_index_13.get(away_team)]
            elif season == '2022-23':
                home_team_series = team_df.iloc[team_index_current.get(home_team)]
                away_team_series = team_df.iloc[team_index_current.get(away_team)]
            else:
                try:
                    home_team_series = team_df.iloc[team_index_14.get(home_team)]
                    away_team_series = team_df.iloc[team_index_14.get(away_team)]
                except Exception as e:
                    print(home_team)
                    raise e
            game = pd.concat([home_team_series, away_team_series.rename(
                index={col:f"{col}.1" for col in team_df.columns.values}
            )])
            games.append(game)

     
odds_con.close()
teams_con.close()
frame = pd.concat(games, ignore_index=True, axis=1)
frame = frame.T
frame['Score_H'] = np.asarray(score_home)
frame['Score_A'] = np.asarray(score_away)
frame['Days-Rest-Home'] = np.asarray(days_rest_home)
frame['Days-Rest-Away'] = np.asarray(days_rest_away)
# fix types
for field in frame.columns.values:
    if 'TEAM_' in field  or 'Date' in field or field not in frame:
        continue
    frame[field] = frame[field].astype(float)
con = sqlite3.connect("../../Data/dataset.sqlite")
frame.to_sql("dataset_2012-23", con, if_exists="replace")
con.close()
