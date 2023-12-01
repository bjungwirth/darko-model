import pandas as pd 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
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
away_team_logs.columns = [f'away_{col}' if col not in ['team_id', 'game_id', 'game_date'] else col for col in away_team_logs.columns]

# Merge the two DataFrames based on 'game_id'
merged_logs = pd.merge(home_team_logs, away_team_logs, on='game_id')

# Calculate pace for each game using the formula
merged_logs['pace'] = 0.5 * (
    (merged_logs['fga'] + 0.4 * merged_logs['fta'] - 1.07 * (merged_logs['oreb'] / (merged_logs['oreb'] + merged_logs['dreb'])) * (merged_logs['fga'] - merged_logs['fgm']) + merged_logs['tov']) +
    (merged_logs['away_fga'] + 0.4 * merged_logs['away_fta'] - 1.07 * (merged_logs['away_oreb'] / (merged_logs['away_oreb'] + merged_logs['away_dreb'])) * (merged_logs['away_fga'] - merged_logs['away_fgm']) + merged_logs['away_tov'])
)

merged_logs['home_point_diff'] = merged_logs['pts'] - merged_logs['away_pts'] 
merged_logs['away_point_diff'] = merged_logs['away_pts'] - merged_logs['pts']

merged_logs = merged_logs[['season','team_id_x', 'team_abbreviation', 'team_id_y', 'away_team_abbreviation',  'game_id', 'game_date_x', 'home_point_diff', 'away_point_diff', 'pace']]

merged_logs.columns = ['season', 'home_team_id', 'home_team_abbreviation', 'away_team_id',  'away_team_abbreviation',  'game_id', 'game_date', 'home_point_diff', 'away_point_diff', 'pace']

df = merged_logs

def exponential_weight(season):
    latest_season = '2023-24'
    latest_year = int(latest_season.split('-')[0])
    season_year = int(season.split('-')[0])
    num_years = latest_year - season_year
    decay_rate = 0.85  # Adjust this rate as needed
    weight = pow(decay_rate, num_years)
    return weight

def linear_weight(season):
    latest_season = '2023-24'
    latest_year = int(latest_season.split('-')[0])
    season_year = int(season.split('-')[0])
    num_years = latest_year - season_year
    weight = max(0.1, 1.0 - 0.1 * num_years)  # Adjust the decrement as needed
    return weight


df['game_date'] = pd.to_datetime(df['game_date'])

ids_abbrevs = df[['home_team_id', 'home_team_abbreviation']].drop_duplicates()
ids_abbrevs_dict = dict(zip(ids_abbrevs['home_team_id'], ids_abbrevs['home_team_abbreviation']))

df.sort_values(by=['game_date'], inplace=True)

df['normalized_home_diff'] = df['home_point_diff'] / df['pace']
df['normalized_away_diff'] = df['away_point_diff'] / df['pace']

# Create separate dataframes for home and away teams
df_home = df[['season', 'game_date', 'home_team_id', 'normalized_home_diff']].rename(columns={'home_team_id': 'team_id', 'normalized_home_diff': 'normalized_diff'})
df_away = df[['season', 'game_date', 'away_team_id', 'normalized_away_diff']].rename(columns={'away_team_id': 'team_id', 'normalized_away_diff': 'normalized_diff'})

df_combined = pd.concat([df_home, df_away])

# Sort df_combined by team and game_date to correctly identify back-to-back games
df_combined.sort_values(by=['team_id', 'game_date'], inplace=True)

# Identify back-to-back games for all teams
df_combined['back_to_back'] = df_combined.groupby('team_id')['game_date'].diff() == pd.Timedelta(days=1)

# Group by team, season, and back-to-back status, then calculate mean normalized point differential
grouped = df_combined.groupby(['team_id', 'season', 'back_to_back'])['normalized_diff'].mean().reset_index()

# Apply weights to seasons
grouped['weight'] = grouped['season'].apply(exponential_weight)  # or linear_weight

# Calculate weighted average normalized differential
grouped['weighted_normalized_diff'] = grouped['normalized_diff'] * grouped['weight']

# Calculate the impact of back-to-back games
back_to_back_impact = grouped.pivot_table(index=['team_id', 'season'], columns='back_to_back', values='weighted_normalized_diff').reset_index()
back_to_back_impact.columns = ['team_id', 'season', 'non_back_to_back', 'back_to_back']

# Calculate the difference between non back-to-back and back-to-back games
back_to_back_impact['impact'] = back_to_back_impact['back_to_back'] - back_to_back_impact['non_back_to_back']

b2b_impact = np.mean(back_to_back_impact['impact'])