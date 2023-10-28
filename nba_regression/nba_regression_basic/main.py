import argparse
from colorama import Fore, Style
import pandas as pd
#import tensorflow as tf
from src.Predict import XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games
from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from sbrscrape import Scoreboard
from datetime import datetime, timedelta
import time


todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2022/scores/00_todays_scores.json'
url = 'https://stats.nba.com/stats/' \
      'leaguedashteamstats?Conference=&' \
      'DateFrom={0}&DateTo={1}' \
      '&Division=&GameScope=&GameSegment=&LastNGames=0&' \
      'LeagueID=00&Location=&MeasureType=Base&Month=0&' \
      'OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&' \
      'PerMode=PerGame&Period=0&PlayerExperience=&' \
      'PlayerPosition=&PlusMinus=N&Rank=N&' \
      'Season=' \
      '&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&' \
      'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

url2 = 'https://stats.nba.com/stats/' \
      'leaguedashteamstats?Conference=&' \
      'DateFrom=&DateTo=' \
      '&Division=&GameScope=&GameSegment=&LastNGames=0&' \
      'LeagueID=00&Location=&MeasureType=Base&Month=0&' \
      'OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&' \
      'PerMode=PerGame&Period=0&PlayerExperience=&' \
      'PlayerPosition=&PlusMinus=N&Rank=N&' \
      'Season=2023-24' \
      '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
      'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='


def createTodaysGames(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    # todo: get the days rest for current games
    home_team_days_rest = []
    away_team_days_rest = []

    for game in games:
        home_team = game[0]
        away_team = game[1]
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        if odds is not None:
            game_odds = odds[home_team + ':' + away_team]
            todays_games_uo.append(game_odds['under_over_odds'])
            
            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])

        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))

            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))
        
        # calculate days rest for both teams
        dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
        schedule_df = pd.read_csv('Data/nba-2023-UTC.csv', parse_dates=['Date'], date_parser=dateparse)
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
        previous_home_games = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date']
        previous_away_games = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date']
        if len(previous_home_games) > 0:
            last_home_date = previous_home_games.iloc[0]
            home_days_off = timedelta(days=1) + datetime.today() - last_home_date
        else:
            home_days_off = timedelta(days=7)
        if len(previous_away_games) > 0:
            last_away_date = previous_away_games.iloc[0]
            away_days_off = timedelta(days=1) + datetime.today() - last_away_date
        else:
            away_days_off = timedelta(days=7)
        # print(f"{away_team} days off: {away_days_off.days} @ {home_team} days off: {home_days_off.days}")

        home_team_days_rest.append(home_days_off.days)
        away_team_days_rest.append(away_days_off.days)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    
    frame_ml = games_data_frame.drop(columns=['GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK',
                                      'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK',
                                      'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK',
                                      'DREB_RANK', 'REB_RANK', 'AST_RANK','TOV_RANK','STL_RANK', 'BLK_RANK', 
                                      'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK','PLUS_MINUS_RANK']) 
    show_ml = frame_ml.drop(columns=['PTS','REB','AST','TOV','STL','BLK','BLKA','PF','PFD','MIN','FGM','FGA','FG_PCT',
                                     'FG3M','FG3A','FG3_PCT','FTM','FTA','FT_PCT','OREB','DREB','TEAM_ID'])
    print(show_ml)

    frame_ml = frame_ml.drop(columns=['TEAM_ID', 'TEAM_NAME'])


    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    odds = {}
    odds = SbrOddsProvider(sportsbook='fanduel').get_odds()
    games = create_todays_games_from_odds(odds)
    if len(games) == 0:
        print("No games found.")
        return
    print(f"----------------- caesars odds data------------------")
    for g in odds.keys():
        home_team, away_team = g.split(":")
        spread = odds[g]['spread']
        away_v = odds[g][away_team]['money_line_odds']
        if(away_v):
            if(away_v<0):
                away_v = -100/away_v + 1
            else:
                away_v = away_v/100 + 1
            away_v = round(away_v,2)
        home_v = odds[g][home_team]['money_line_odds']
        if(home_v):
            if(home_v<0):
                home_v = -100/home_v + 1
            else:
                home_v = home_v/100 + 1
            home_v = round(home_v,2)
        if(spread and spread>0):
            plus_str = '+'
        else:
            plus_str = ''
        print(f"{home_team} ({home_v}) @ {away_team} ({away_v}) -> {plus_str}{spread}")
    t = datetime.today()
    t_from = (t+timedelta(days=-60)).strftime('%m/%d/%Y')
    t_to = t.strftime('%m/%d/%Y')
    data_p = get_json_data(url.format(t_from, t_to))
    
    record_p = data_p[0].get('rowSet')
    data_r = get_json_data(url2)
    record_r = data_r[0].get('rowSet')
    length = len(record_r[0])
    for r in record_r:
        for p in record_p:
            if(p[0] == r[0]):
                for i in range(0,length-1):
                    r[i] = p[i]
    df = pd.DataFrame(data=record_r, columns=data_r[0].get('headers')) 
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)
    
    print("---------------XGBoost Model Predictions---------------")
    XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-now', action='store_true', help='Sportsbook to fetch from.')
    args = parser.parse_args()
    main()
