import requests
from bs4 import BeautifulSoup
import pandas as pd 

url = 'https://apanalytics.shinyapps.io/DARKO/_w_e782c2f9/session/176bd1a4b77c731ef8cc273123729f31/download/download_talent?w=e782c2f9'
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

csv_data = soup.text
# Split the CSV data into lines
csv_lines = csv_data.split('\n')

# Remove the header line if needed
header = csv_lines[0].strip()
csv_lines = csv_lines[1:]

darko = pd.DataFrame([line.split(',') for line in csv_lines[1:]], columns=csv_lines[0].split(','))

darko = darko.dropna()

darko.to_csv('../data/darko.csv', index=False)