import pandas as pd
import pickle
import streamlit as st

def getData():
    with open('outfield.pkl', 'rb') as file:
        player_df = pd.DataFrame(pickle.load(file))
    with open('player_ID.pickle', 'rb') as file:
        player_ID = pickle.load(file)
    with open('engine.pickle', 'rb') as file:
        engine = pickle.load(file)
    
    with open('gk.pkl', 'rb') as file:
        gk_df = pd.DataFrame(pickle.load(file))
    with open('gk_ID.pickle', 'rb') as file:
        gk_ID = pickle.load(file)
    with open('gk_engine.pickle', 'rb') as file:
        gk_engine = pickle.load(file)
    st.write('Columns in player_df: ', player_df.columns)
    st.write('Columns in gk_df: ', gk_df.columns)

    return player_df, player_ID, engine, gk_df, gk_ID, gk_engine

def getRecommendations(df, ID, metric, df_type, league, comparison, query, count=5):
    if df_type == 'outfield players':
        df_res = df.iloc[:, [1, 3, 5, 6, 11,-1]].copy()
    else:
        df_res = df.iloc[:, [1, 3, 5, 6, 11]].copy()
    
    df_res['Player'] = list(ID.keys())
    df_res.insert(1, 'Similarity', metric)
    df_res = df_res.sort_values(by=['Similarity'], ascending=False)
    metric = [str(num) + '%' for num in df_res['Similarity']]
    df_res['Similarity'] = metric
    df_res = df_res.iloc[1:, :]

    if comparison == 'Same position' and df_type == 'outfield players':
        q_pos = list(df[df['Player'] == query.split(' (')[0]]['Pos'])[0]
        df_res = df_res[df_res['Pos'] == q_pos]

    if league != 'All':
        df_res = df_res[df_res['Comp'] == league]

    df_res = df_res.head(count)
    return df_res

st.title('Football Player Recommender')
st.write('A Web App to recommend football players who play similar to your favorite players!')

player_df, player_ID, engine, gk_df, gk_ID, gk_engine = getData()
player_type = st.sidebar.radio('Player type', ['Outfield players', 'Goal Keepers'])

if player_type == 'Outfield players':
    df, ID, engine = player_df, player_ID, engine
else:
    df, ID, engine = gk_df, gk_ID, gk_engine

st.sidebar.markdown('### Input player info')
players = list(ID.keys())
query = st.sidebar.selectbox('Select a player', players)
count = st.sidebar.slider('How many similar players do you want?', min_value=1, max_value=10, value=5)
comparison = st.sidebar.selectbox('Comparison', ['All positions', 'Same position'])
league = st.sidebar.selectbox('League', ['All', 'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'])

metric = engine[query]

result = getRecommendations(df, ID, metric, player_type.lower(), league, comparison, query, count)
st.markdown('### Here are players who play similar to {}'.format(query.split(' (')[0]))
st.dataframe(result)
