import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import decomposition
from scipy.spatial import distance
from tqdm import tqdm
import pickle

redundant = ['Rk','Player','Nation','Pos','Squad','Comp','Age','Born','90s','Matches']

def load_data():
    df = pd.read_csv('general.csv', header=0).drop(['Rk','Matches'], axis=1)
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

    grand = pd.concat([df , shooting, passing, passing_types, gca, defense, possession, misc], axis=1)
    df = grand[grand['90s'] >= 3]
    df = df[df['Pos'] != 'GK'].reset_index(drop=True)
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

with open('player_ID.pickle', 'wb') as file:
    pickle.dump(player_ID, file)

stats = df.iloc[:, 12:-1]
data = StandardScaler().fit_transform(stats)

# Convert stats to a numpy array after standardization
stats = np.array(data)

model = TSNE(n_components=2, perplexity=30, random_state=0)
tsne_data = model.fit_transform(stats)
tsne_data = np.vstack((tsne_data.T, df['Pos'])).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension 1", "Dimension 2", "Positions"))

pca = decomposition.PCA()
pca.n_components = 142
pca_data = pca.fit_transform(stats)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_explained)

# The number of components should be the same as in the PCA
stats = pca_data[:, :142]

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

with open('outfield.pkl', 'wb') as file:
    pickle.dump(stats, file)

# Save column names for future reference
with open('column_names.pkl', 'wb') as file:
    pickle.dump(list(range(142)), file)
