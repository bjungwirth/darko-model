import pandas as pd 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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

merged_logs = merged_logs[['season','team_id_x', 'team_abbreviation', 'team_id_y', 'away_team_abbreviation',  'game_id', 'home_point_diff', 'away_point_diff', 'pace']]

merged_logs.columns = ['season', 'home_team_id', 'home_team_abbreviation', 'away_team_id',  'away_team_abbreviation',  'game_id', 'home_point_diff', 'away_point_diff', 'pace']

df = merged_logs

ids_abbrevs = df[['home_team_id', 'home_team_abbreviation']].drop_duplicates()
ids_abbrevs_dict = dict(zip(ids_abbrevs['home_team_id'], ids_abbrevs['home_team_abbreviation']))

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


# Normalizing point differentials by pace
df['normalized_home_diff'] = df['home_point_diff'] / df['pace']
df['normalized_away_diff'] = df['away_point_diff'] / df['pace']

# Split into home and away dataframes
df_home = df[['season', 'home_team_id', 'normalized_home_diff']].rename(columns={'home_team_id': 'team_id', 'normalized_home_diff': 'normalized_diff'})
df_away = df[['season', 'away_team_id', 'normalized_away_diff']].rename(columns={'away_team_id': 'team_id', 'normalized_away_diff': 'normalized_diff'})

df_home['is_home'] = True
df_away['is_home'] = False

# Combine home and away dataframes
df_combined = pd.concat([df_home, df_away])

# Group by team, season, and home/away status, then calculate mean normalized differential
grouped = df_combined.groupby(['team_id', 'season', 'is_home'])['normalized_diff'].mean().reset_index()

# Apply linear weights to seasons
grouped['linear_weight'] = grouped['season'].apply(linear_weight)

# Apply exponential weights to seasons
grouped['exponential_weight'] = grouped['season'].apply(exponential_weight)

# Calculate weighted average differential using linear weights
grouped['weighted_diff_linear'] = grouped['normalized_diff'] * grouped['linear_weight']

# Calculate weighted average differential using exponential weights
grouped['weighted_diff_exponential'] = grouped['normalized_diff'] * grouped['exponential_weight']

# Calculate home court advantage using linear weights
home_court_advantage_linear = grouped.groupby('team_id').apply(lambda x: x[x['is_home']]['weighted_diff_linear'].mean() - x[~x['is_home']]['weighted_diff_linear'].mean()).reset_index(name='home_court_advantage_linear')

# Calculate home court advantage using exponential weights
home_court_advantage_exponential = grouped.groupby('team_id').apply(lambda x: x[x['is_home']]['weighted_diff_exponential'].mean() - x[~x['is_home']]['weighted_diff_exponential'].mean()).reset_index(name='home_court_advantage_exponential')

home_court_advantage = home_court_advantage_linear.merge(home_court_advantage_exponential, on='team_id')

home_court_advantage['team_abbreviation'] = home_court_advantage['team_id'].map(ids_abbrevs_dict)

home_court_advantage.to_csv('../data/home_court_advantage.csv', index=False)