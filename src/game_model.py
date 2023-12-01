import pandas as pd 
import requests
from bs4 import BeautifulSoup
import pandas as pd 
from io import StringIO
import re
from nba_api.stats.endpoints import playerestimatedmetrics, teamestimatedmetrics, leaguegamelog
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType
import datetime as dt 
import pickle
import numpy as np
import gspread
from google.oauth2 import service_account

b2b_impact = -0.014643513054999802
hca = pd.read_csv('../data/home_court_advantage.csv')
hca = hca[['team_abbreviation', 'home_court_advantage_exponential']]
hca = hca.rename(columns={'home_court_advantage_exponential':'est_hca_effect'})

s = '2023-24'
game_log = leaguegamelog.LeagueGameLog(season=s, season_type_all_star='Regular Season').get_data_frames()[0]
game_log['season'] = s
game_log.columns = game_log.columns.str.lower()
game_log['game_date'] = pd.to_datetime(game_log['game_date'])

last_played_game = game_log.groupby('team_abbreviation')['game_date'].max().reset_index()

current_date = dt.datetime.today().date()
last_played_game['days_since_last_game'] = current_date - last_played_game['game_date'].dt.date
last_played_game['is_b2b'] = last_played_game['days_since_last_game'] == dt.timedelta(days=1)
last_played_game = last_played_game[['team_abbreviation', 'is_b2b']]

# Load your service account JSON credentials
credentials = service_account.Credentials.from_service_account_file('../sacred-atom-348720-603c9c70cde5.json', scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])

# Authenticate with Google Sheets
gc = gspread.Client(auth=credentials)

def preprocess_player_name(name):
    cleaned_name = re.sub(r'[^\w\s]', ' ', name)  # Remove punctuation and spaces
    cleaned_name = cleaned_name.lower().strip()  # Convert to lowercase
    return cleaned_name

#convert darko team names to abbreviations
team_abbrevs_dict = {
    'Atlanta Hawks' : 'ATL',
    'Boston Celtics' : 'BOS',
    'Brooklyn Nets' : 'BKN',
    'Charlotte Hornets' : 'CHA',
    'Chicago Bulls' : 'CHI',
    'Cleveland Cavaliers' : 'CLE',
    'Dallas Mavericks' : 'DAL',
    'Denver Nuggets' : 'DEN',
    'Detroit Pistons' : 'DET',
    'Golden State Warriors' : 'GSW',
    'Houston Rockets' : 'HOU',
    'Indiana Pacers' : 'IND',
    'Los Angeles Clippers' : 'LAC',
    'Los Angeles Lakers' : 'LAL',
    'Memphis Grizzlies' : 'MEM',
    'Miami Heat' : 'MIA',
    'Milwaukee Bucks' : 'MIL',
    'Minnesota Timberwolves' : 'MIN',
    'New Orleans Pelicans' : 'NOP',
    'New York Knicks' : 'NYK',
    'Oklahoma City Thunder' : 'OKC',
    'Orlando Magic' : 'ORL',
    'Philadelphia 76ers' : 'PHI',
    'Phoenix Suns' : 'PHX',
    'Portland Trail Blazers' : 'POR',
    'Sacramento Kings' : 'SAC',
    'San Antonio Spurs' : 'SAS',
    'Toronto Raptors' : 'TOR',
    'Utah Jazz' : 'UTA',
    'Washington Wizards' : 'WAS',
    'LA Clippers' : 'LAC',
}

player_pace = playerestimatedmetrics.PlayerEstimatedMetrics(season=Season.default, season_type=SeasonType.default).get_data_frames()[0]
player_pace.columns = player_pace.columns.str.lower()

player_pace = player_pace[['player_id', 'min', 'e_pace']]

team_pace = teamestimatedmetrics.TeamEstimatedMetrics(season=Season.default, season_type=SeasonType.default).get_data_frames()[0]
team_pace.columns = team_pace.columns.str.lower()

team_pace = team_pace[['team_name', 'team_id', 'min', 'e_pace']]

team_pace['team_name'] = team_pace['team_name'].replace(team_abbrevs_dict)

#calc team league average pace
lg_avg_team_pace = team_pace['e_pace'].mean()
lg_avg_player_pace = player_pace['e_pace'].mean()

url = 'https://docs.google.com/spreadsheets/d/1mhwOLqPu2F9026EQiVxFPIN1t9RGafGpl-dokaIsm9c/export?format=csv&gid=1064086941'

darko = pd.read_csv(url)

# Drop rows with missing values (NaN)
darko = darko.dropna(subset='DPM')
darko = darko.merge(player_pace, left_on='NBA ID', right_on='player_id', how='left')

#probably should jsut import player ids from nba api 
stok_name_changes = {
    #'Nicolas Claxton' : 'Nic Claxton',
    'Bones Hyland' : "Nah'Shon Hyland",
    'Craig Porter Jr.' : 'Craig Porter',
    'PJ Washington' : 'P.J. Washington',
    'Nic Claxton' : 'Nicolas Claxton',
    'AJ Green' : 'A.J. Green',
    'KJ Martin': 'Kenyon Martin Jr.',
    'John Butler': 'John Butler Jr.'
}


projections = pd.read_csv('../data/projs.csv')

min_projs = projections[['Name', 'Team', 'Proj minutes']]

min_projs['Name'] = min_projs['Name'].replace(stok_name_changes)

min_projs['Name'] = min_projs['Name'].apply(preprocess_player_name)

min_projs = min_projs.merge(team_pace, left_on='Team', right_on='team_name', suffixes=('', '_team'),how='left')


# Apply the preprocess_player_name function to the 'Name' column in min_projs using .loc
# Apply the preprocess_player_name function to the 'Player' column in darko using .loc
darko['Player'] = darko['Player Name'].apply(preprocess_player_name)

darko = darko[['Player',  'DPM', 'e_pace', 'min',]]

darko = darko.rename(columns={'min':'avg_min'})

avg_dpm = darko['DPM'].mean()

# Perform a left join between 'min_projs' and 'darko' on the 'Name' column
missing_players = min_projs.merge(darko, left_on=['Name'], right_on=['Player'], how='left')

df = min_projs.merge(darko, left_on=['Name'], right_on=['Player'], how='left')

df = df.drop(columns=['Player'])

df = df.rename(columns={'Proj minutes':'proj_min'})
matchups = projections['Matchup'].drop_duplicates().str.split('@', expand=True)
matchups['home'] = matchups[1].str.strip()
matchups['away'] = matchups[0].str.strip()
matchups = matchups.drop(columns=[0,1])

df.columns = df.columns.str.lower()

with open('../models/pace_model.pkl', 'rb') as file:
    pace_model = pickle.load(file)

matchups = matchups.merge(team_pace[['team_name', 'e_pace']], left_on='home', right_on='team_name')
matchups = matchups.merge(team_pace[['team_name', 'e_pace']], left_on='away', right_on='team_name')

matchups = matchups.drop(columns=['team_name_x', 'team_name_y'])
matchups.columns = ['home', 'away', 'home_team_avg_pace', 'away_team_avg_pace']
matchups['lg_avg_pace'] = lg_avg_team_pace

X_new = matchups[['home_team_avg_pace', 'away_team_avg_pace', 'lg_avg_pace']]
predicted_pace = pace_model.predict(X_new)

matchups['pace_pred'] = predicted_pace

print(matchups[['home', 'away', 'pace_pred']])

home_preds = matchups[['home', 'pace_pred']]
home_preds.columns = ['team', 'pace_pred']
away_preds = matchups[['away', 'pace_pred']]
away_preds.columns = ['team', 'pace_pred']
all_preds = pd.concat([home_preds, away_preds])

player_dpm = df[['name','team','proj_min','dpm']]

player_dpm = player_dpm.merge(all_preds, left_on='team', right_on='team')

# Calculate per-possession impact
player_dpm['impact_per_possession'] = player_dpm['dpm'] / 100

# Adjust for the projected pace of the game
player_dpm['impact_per_48'] = player_dpm['impact_per_possession'] * player_dpm['pace_pred']

# Scale the impact based on projected minutes
# This gives the total impact for the projected minutes each player is expected to play
player_dpm['total_impact'] = player_dpm['impact_per_48'] * (player_dpm['proj_min'] / 48)

# Aggregate this impact for each team
team_impact = player_dpm.groupby('team')['total_impact'].sum().reset_index()

matchups = matchups.merge(team_impact, left_on='home', right_on='team')
matchups = matchups.merge(team_impact, left_on='away', right_on='team')
matchups = matchups.drop(columns=['team_x', 'team_y'])

matchups.columns = ['home', 'away', 'home_team_avg_pace', 'away_team_avg_pace', 'lg_avg_pace', 'pace_pred', 'home_team_impact', 'away_team_impact']
matchups = matchups[['home', 'away', 'pace_pred', 'home_team_impact', 'away_team_impact']]

matchups = matchups.merge(hca, left_on='home', right_on='team_abbreviation')
matchups = matchups.drop(columns=['team_abbreviation'])
matchups['est_hca_effect'] = matchups['est_hca_effect'] * matchups['pace_pred']
matchups['raw_spread'] = matchups['home_team_impact'] - matchups['away_team_impact']
matchups = matchups.merge(last_played_game, left_on='home', right_on='team_abbreviation').drop(columns=['team_abbreviation']).merge(last_played_game, left_on='away', right_on='team_abbreviation').drop(columns=['team_abbreviation'])

matchups.columns = ['home', 'away', 'pace_pred', 'home_team_impact', 'away_team_impact', 'est_hca_effect', 'raw_spread', 'home_b2b', 'away_b2b']

matchups['proj_spread'] = (matchups['home_team_impact']+matchups['est_hca_effect']-(matchups['home_b2b']*b2b_impact)  - (matchups['away_team_impact']-(matchups['away_b2b']*b2b_impact)))

matchups['favorite'] = np.where(matchups['proj_spread'] > 0, matchups['home'], matchups['away'])

#get odds
# Replace 'YOUR_API_KEY' with your actual API key from The Odds API
api_key = '701edd25671e94ae93f0d2bf868ae491'

# Set up the API endpoint and parameters
url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
params = {
    'apiKey': api_key,
    'regions': 'us',  # Assuming you're interested in US bookmakers
    'markets': 'spreads',  # To fetch the spread odds
    'bookmakers': 'pinnacle,draftkings,fanduel,betonline,bookmaker,bovada'
}

# Make the API request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    odds_data = response.json()
    # Process the data as needed
else:
    print(f"Error fetching data: {response.status_code}")


def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else: 
        return int(-100 / (decimal_odds - 1))

# Initialize an empty DataFrame to store the final data
final_df = pd.DataFrame()

# Loop through each game and bookmaker, constructing the DataFrame
for game in odds_data:
    home_team = game['home_team']
    away_team = game['away_team']

    # Dictionary to hold the data for this game
    game_data = {'home_team': home_team, 'away_team': away_team}

    # Extract data for each bookmaker
    for bookmaker in game['bookmakers']:
        bookmaker_key = bookmaker['key']
        for market in bookmaker['markets']:
            if market['key'] == 'spreads':
                # Convert decimal odds to American odds
                home_price_american = decimal_to_american(market['outcomes'][0]['price'])
                away_price_american = decimal_to_american(market['outcomes'][1]['price'])

                # Add the data to the game_data dictionary
                game_data[f'{bookmaker_key}_home_spread'] = market['outcomes'][0]['point']
                game_data[f'{bookmaker_key}_home_price'] = home_price_american
                game_data[f'{bookmaker_key}_away_spread'] = market['outcomes'][1]['point']
                game_data[f'{bookmaker_key}_away_price'] = away_price_american

    # Convert the game data to a DataFrame and append it to the final DataFrame
    game_df = pd.DataFrame([game_data])
    final_df = pd.concat([final_df, game_df], ignore_index=True)

# Display the final DataFrame
print(final_df)

# Merge the final_df DataFrame with the matchups DataFrame
final_df['home_team'] = final_df['home_team'].replace(team_abbrevs_dict)
final_df['away_team'] = final_df['away_team'].replace(team_abbrevs_dict)

matchups = matchups.merge(final_df, left_on=['home', 'away'], right_on=['home_team', 'away_team'])

# Calculate 'proj_spread_diff' conditionally using 'pinnacle_away_spread' and 'draftkings_away_spread'
matchups['proj_spread_diff'] = np.where(matchups['pinnacle_away_spread'].notnull(), 
                                         matchups['pinnacle_away_spread'] - matchups['proj_spread'], 
                                         matchups['draftkings_away_spread'] - matchups['proj_spread'])

matchups['proj_spread'] = matchups['proj_spread']*-1
matchups = matchups[['home', 'away', 'pace_pred', 'home_team_impact', 'away_team_impact',
       'raw_spread', 'est_hca_effect',  'home_b2b', 'away_b2b', 'proj_spread', 'favorite', 'proj_spread_diff', 'draftkings_home_spread', 'draftkings_home_price',
       'draftkings_away_spread', 'draftkings_away_price',
       'fanduel_home_spread', 'fanduel_home_price', 'fanduel_away_spread',
       'fanduel_away_price', 'pinnacle_home_spread', 'pinnacle_home_price',
       'pinnacle_away_spread', 'pinnacle_away_price', 'bovada_home_spread',
       'bovada_home_price', 'bovada_away_spread', 'bovada_away_price']]

sheet = gc.open('NBA preds')
worksheet = sheet.get_worksheet(0)

# Assuming 'matchups_df' is your DataFrame
from gspread_pandas import Spread

# Open an existing Google Sheet by title
spread = Spread('NBA preds', creds=credentials)

# Select a specific worksheet by title
worksheet = spread.sheet_to_df(sheet='Sheet1')

# Write your DataFrame to the selected worksheet
spread.df_to_sheet(matchups, sheet='Sheet1', start='A1', index=False, replace=True)
#matchups.to_csv('../output/matchups.csv', index=False)

