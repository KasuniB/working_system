import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import decomposition
from scipy.spatial import distance
from tqdm import tqdm
import pickle

redundant = ['Rk','Player','Nation','Pos','Squad','Comp','Age','Born','90s','Matches']

general = pd.read_csv('general.csv', header=0).drop(['Rk','Matches'], axis=1)
shooting = pd.read_csv('shooting.csv', header=0).drop(redundant, axis=1)
passing = pd.read_csv('passing.csv', header=0).drop(redundant, axis=1)
passing_types = pd.read_csv('passing_types.csv', header=0).drop(redundant, axis=1)
gca = pd.read_csv('gca.csv', header=0).drop(redundant, axis=1)
defense = pd.read_csv('defense.csv', header=0).drop(redundant, axis=1)
possession = pd.read_csv('possession.csv', header=0).drop(redundant, axis=1)
misc = pd.read_csv('misc.csv', header=0).drop(redundant, axis=1)

def renameColumns(table_no, df):
    num = str(table_no) + "_"
    return df.rename(columns=lambda x: num + x)

shooting = renameColumns(2, shooting)
passing = renameColumns(3, passing)
passing_types = renameColumns(4, passing_types)
gca = renameColumns(5, gca)
defense = renameColumns(6, defense)
possession = renameColumns(7, possession)
misc = renameColumns(8, misc)

grand = pd.concat([general , shooting, passing, passing_types, gca, defense, possession, misc], axis=1)
print(grand.head())
df = grand[grand['90s'] >= 3]
df = df[df['Pos'] != 'GK'].reset_index()
df['Comp'] = df['Comp'].str.split(' ', expand=True, n=1)[1]

with open('outfield.pkl', 'wb') as file:
    pickle.dump(df, file)

players = []
for idx in range(len(df)):
    players.append(df['Player'][idx] + '({})'.format(df['Squad'][idx]))
player_ID = dict(zip(players, np.arange(len(players))))

with open('player_ID.pickle', 'wb') as file:
    pickle.dump(player_ID, file)

print(df)

stats = df.iloc[:, 11:-1]
labels = df['Pos']
data = StandardScaler().fit_transform(stats)

pca = decomposition.PCA()
pca.n_components = 150
pca_data = pca.fit_transform(data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_explained)

# The number of components should be the same as in the PCA
stats = pca_data[:, :150]

def getStats(name):
    idx = player_ID[name]
    return stats[idx, :]

def similarity(player1, player2):
   return 1- distance.cosine(getStats(player1),getStats(player2))

def normalize(array):
   return np.array([round(num, 2) for num in (array - min(array))*100/(max(array)-min(array))])

engine = {}
for query in tqdm(players):
    metric = []
    for player in players:
        value = similarity(query, player)
        metric.append(value)
    metric = normalize(metric)
    engine[query] = metric

with open('engine.pickle', 'wb') as file:
    pickle.dump(engine, file)

