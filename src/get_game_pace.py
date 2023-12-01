import pandas as pd 

from nba_api.stats.endpoints import teamestimatedmetrics
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType
from nba_api.stats.endpoints import teamestimatedmetrics
from nba_api.stats.endpoints import leaguegamelog

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

seasons = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21','2021-22', '2022-23', '2023-24']

game_logs = []
metrics = []
for s in seasons:
    print(s)
    game_log = leaguegamelog.LeagueGameLog(season=s, season_type_all_star='Regular Season').get_data_frames()[0]
    game_log['season'] = s
    game_log.columns = game_log.columns.str.lower()
    game_logs.append(game_log)
    metric = teamestimatedmetrics.TeamEstimatedMetrics(season=s, season_type=SeasonType.default).get_data_frames()[0]
    metric = metric[['TEAM_ID', 'E_PACE']]
    metric.columns = metric.columns.str.lower()
    metric['season'] = s
    metrics.append(metric)

game_logs = pd.concat(game_logs)
metrics = pd.concat(metrics)

#calcualte pace for each game
# Split the DataFrame into two separate DataFrames (home team and away team)
home_team_logs = game_logs[game_logs['matchup'].str.contains('vs.')]
away_team_logs = game_logs[game_logs['matchup'].str.contains('@')]

# Rename columns in away_team_logs to indicate it's the opponent
away_team_logs.columns = [f'opp_{col}' if col not in ['team_id', 'game_id', 'game_date'] else col for col in away_team_logs.columns]

# Merge the two DataFrames based on 'game_id'
merged_logs = pd.merge(home_team_logs, away_team_logs, on='game_id')

# Calculate pace for each game using the formula
merged_logs['pace'] = 0.5 * (
    (merged_logs['fga'] + 0.4 * merged_logs['fta'] - 1.07 * (merged_logs['oreb'] / (merged_logs['oreb'] + merged_logs['dreb'])) * (merged_logs['fga'] - merged_logs['fgm']) + merged_logs['tov']) +
    (merged_logs['opp_fga'] + 0.4 * merged_logs['opp_fta'] - 1.07 * (merged_logs['opp_oreb'] / (merged_logs['opp_oreb'] + merged_logs['opp_dreb'])) * (merged_logs['opp_fga'] - merged_logs['opp_fgm']) + merged_logs['opp_tov'])
)

merged_logs = merged_logs[['season','team_id_x', 'team_id_y', 'game_id', 'pace']]

merged_logs.columns = ['season', 'home_team_id', 'away_team_id', 'game_id', 'pace']

metrics = metrics.merge(merged_logs[['team_id', 'game_id', 'pace']], on=['team_id', 'game_id'])

df = merged_logs.merge(metrics, left_on=['home_team_id','season'], right_on=['team_id','season'])

df = df.merge(metrics, left_on=['away_team_id','season'], right_on=['team_id','season'])

df = df[['season', 'home_team_id', 'away_team_id', 'game_id', 'pace', 'e_pace_x', 'e_pace_y']]

df.columns = ['season', 'home_team_id', 'away_team_id', 'game_id', 'game_pace', 'home_team_avg_pace', 'away_team_avg_pace']

#calc league average pace per season
lg_avg_pace = metrics.groupby('season')['e_pace'].mean().reset_index()

df = df.merge(lg_avg_pace, on='season')

df.rename(columns={'e_pace':'lg_avg_pace'}, inplace=True)

df.to_csv('game_paces.csv', index=False)

X = df[['home_team_avg_pace', 'away_team_avg_pace', 'lg_avg_pace']]
y = df['game_pace']
