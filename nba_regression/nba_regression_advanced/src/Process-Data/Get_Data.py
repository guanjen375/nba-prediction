import os
import random
import sqlite3
import time
import sys
from datetime import date, datetime, timedelta

from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Utils.tools import get_json_data, to_data_frame
url = 'https://stats.nba.com/stats/' \
      'leaguedashlineups?Conference=&' \
      'GroupQuantity=5' \
      'DateFrom={0}&DateTo={1}' \
      '&Division=&GameScope=&GameSegment=&LastNGames=0&' \
      'LeagueID=00&Location=&MeasureType=Base&Month=0&' \
      'OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&' \
      'PerMode=PerGame&Period=0&PlayerExperience=&' \
      'PlayerPosition=&PlusMinus=N&Rank=N&' \
      'Season={2}' \
      '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
      'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

year = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

season = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
          "2017-18", "2018-19", "2019-20"]
#2020-2021:
#year = [2020,2021]
#season = ["2020-21"]
#month = [12,1,2,3,4,5,6,7,8,9,10,11]

#2021-2023:
#year = [2021,2022,2023]
#season = ["2021-22","2022-23"]
#month = [10,11,12,1,2,3,4,5,6,7,8,9]


month = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

begin_year_pointer = year[0]
end_year_pointer = year[0]
count = 0

con = sqlite3.connect("../../Data/teams_Special.sqlite")

for season1 in tqdm(season):
    for month1 in tqdm(month):
        if month1 == 1:
            count += 1
            end_year_pointer = year[count]
        for day1 in tqdm(days):
            if month1 in [4,6,9,11] and day1 > 30:
                continue
            if month1 == 2 and day1 > 28:
                continue
            if end_year_pointer == datetime.now().year:
                if month1 == datetime.now().month and day1 > datetime.now().day:
                    continue
                if month1 > datetime.now().month:
                    continue
            t = date(end_year_pointer,month1,day1)
            t_from = (t+timedelta(days=-20)).strftime('%m/%d/%Y')
            t_to = t.strftime('%m/%d/%Y')
            general_data = get_json_data(url.format(t_from, t_to ,season1))
            #general_data = get_json_data(url.format(month1, day1, begin_year_pointer, end_year_pointer, season1))
            general_df = to_data_frame(general_data)
            real_date = date(year=end_year_pointer, month=month1, day=day1) + timedelta(days=1)
            general_df['Date'] = str(real_date)
            print(general_df)
            x = str(real_date).split('-')
            general_df.to_sql(f"teams_{season1}-{str(int(x[1]))}-{str(int(x[2]))}", con, if_exists="replace")
            
            time.sleep(0.5)
    begin_year_pointer = year[count]

con.close()