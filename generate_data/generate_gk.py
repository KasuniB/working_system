import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy.spatial import distance
from tqdm import tqdm
import pickle

redundant = ['Rk','Player','Nation','Pos','Squad','Comp','Age','Born','90s','Matches']

def load_data():
    df = pd.read_csv('goalkeeping.csv', header=0).drop(['Rk','Matches'], axis=1)
    advanced_goalkeeping = pd.read_csv('advanced_goalkeeping.csv', header=0).drop(redundant, axis=1)
    
    def renameColumns(table_no, df):
        num = str(table_no) + "_"
        return df.rename(columns=lambda x: num + x)

    advanced_goalkeeping = renameColumns(2, advanced_goalkeeping)
    grand = pd.concat([df , advanced_goalkeeping], axis=1)
    df = grand[grand['90s'] >= 3]
    df.loc[:, 'Player'] = df['Player'].str.split('\\', expand=True)[0]
    df.loc[:, 'Comp'] = df['Comp'].str.split(' ', expand=True, n=1)[1]
    return df

def similarity(player1, player2):
    p1 = stats[player_ID[player1], :]
    p2 = stats[player_ID[player2], :]
    sim = 100 - distance.cosine(p1, p2) * 100
    return round(sim, 2)

def normalize(metric):
    min_metric = min(metric)
    max_metric = max(metric)
    return [(m - min_metric) / (max_metric - min_metric) * 100 for m in metric]

df = load_data()
players = []
for idx, row in df.iterrows():
    players.append(row['Player'] + '({})'.format(row['Squad']))

player_ID = dict(zip(players, np.arange(len(players))))
with open('gk_ID.pickle', 'wb') as file:
    pickle.dump(player_ID, file)

stats = df.iloc[:, 12:-1]
data = StandardScaler().fit_transform(stats)
# Convert stats to a numpy array after standardization
stats = np.array(data)

pca = decomposition.PCA()
pca.n_components = 38
pca_data = pca.fit_transform(data)

# The number of components should be the same as in the PCA
stats = pca_data[:, :38]
with open('column_names.pkl', 'wb') as file:
    pickle.dump(list(range(38)), file)

engine = {}
for query in tqdm(players):
    metric = []
    for player in players:
        value = similarity(query, player)
        metric.append(value)
    metric = normalize(metric)
    engine[query] = metric

with open('gk_engine.pickle', 'wb') as file:
    pickle.dump(engine, file)

with open('gk.pkl', 'wb') as file:
    pickle.dump(stats, file)
