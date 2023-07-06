import pandas as pd
import requests
from bs4 import BeautifulSoup

url = 'https://fbref.com/en/comps/Big5/playingtime/players/Big-5-European-Leagues-Stats'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'lxml')

html_content = str(soup)

df = pd.read_html(html_content)

for idx, table in enumerate(df):
    print("***************************")
    print(idx)
    print(table)

    # Save the table to a CSV file
    table.to_csv(f'playing_time.csv', index=False)
