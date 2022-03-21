import pickle

import matplotlib.pyplot as plt
import numpy as np
import altair
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
from my_functions import functions

warnings.filterwarnings("ignore")


@st.cache(allow_output_mutation=True)
def get_data(filename):
    df = pd.read_csv(filename)

    try:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    except KeyError:
        pass

    return df


@st.cache(allow_output_mutation=True)
def get_best_model(filename):
    with open(filename, 'rb') as file:
        my_best_model = pickle.load(file)
    return my_best_model


st.set_page_config(layout="wide")
st.maxUploadSize = 500

header = st.container()
intro = st.container()
show_results = st.container()

# df_nn = get_data('csv_files/df_nn.csv')
# my_nn = get_best_model('notebooks/nn_model.pkl')
original_test = get_data('csv_files/original_test.csv')
original_train = get_data('csv_files/original_train.csv')

columns_test = ['SK_ID_CURR', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'AMT_ANNUITY', 'AMT_CREDIT']


def train_nn(df_train, cols, n_neighbors=4):
    # Collecting data
    df_nn = df_train[cols]
    # Collecting target
    df_nn['TARGET'] = df_train['TARGET']
    # Get rid of NAN
    df_nn.dropna(subset=cols, inplace=True)

    # Standardisation
    std = StandardScaler()
    df_std = std.fit_transform(df_nn[cols])
    df_std = pd.DataFrame(df_std,
                          index=df_nn.index,
                          columns=[cols])

    # Train model
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(df_std)

    return nn, df_nn, std


my_nn, df_nn, std = train_nn(original_train, cols=columns_test)


def get_kneighbors(df_test, df_train, trained_model, cols, ID_client):
    # Collecting index of requested ID_client
    my_index = df_test[cols][df_test[cols]['SK_ID_CURR'] == ID_client].index[0]

    # Standaridation of data to predict
    client_list = std.transform(df_test[cols])

    # Prediction
    distance, voisins = trained_model.kneighbors([client_list[my_index]])
    voisins = voisins[0]

    voisins_table = pd.DataFrame()
    # Create df with k nearest neighbours
    for v in range(len(voisins)):
        print(v)
        voisins_table[v] = df_train.iloc[voisins[v]]

    return voisins_table


with header:
    st.title("Dashboard interactif")
    st.text("Ce dashboard a pour but d'aider les chargés de relation client afin qu'ils puissent à la fois "
            "expliquer de façon la plus transparente possible les décisions d’octroi de crédit, "
            "mais également permettre à leurs\n"
            "clients de disposer de leurs informations personnelles et de les explorer facilement. \n")

with st.spinner('Chargement des données clients'):
    test = get_data('csv_files/vue_generale_test.csv')
    train = get_data('csv_files/vue_generale_train.csv')

with st.spinner('Chargement du modèle'):
    model = get_best_model('LGBM_model.pkl')

with st.spinner('Chargement des prédictions'):
    predictions = get_data('csv_files/submission.csv')

# st.dataframe(test)
# st.dataframe(predictions)

col1, col2 = st.columns([6, 10])
with col1:
    st.markdown("Merci d'entrer un identifiant client :")
    identifiant = st.selectbox(label='', options=test['Id_client'])
    df_id = predictions.loc[predictions['ID'] == identifiant]

    st.text('Prédictions de remboursement')
    st.dataframe(df_id)

    percent = np.round(df_id['Prediction'].iloc[0], 3) * 100
    st.text(f"Ce client a {percent} % de chances de rembourser")

    if percent >= 50:
        st.write("Prêt accordé ! ✔")

    if percent < 50:
        st.write("Prêt refusé ! ❌")

with col2:
    st.markdown("Vue générale")
    st.dataframe(test.loc[test['Id_client'] == identifiant].set_index('Id_client'))

    st.text("Clients similaires dans l'historique de la base de données")

    neighbours = get_kneighbors(df_test=original_test, df_train=df_nn, trained_model=my_nn, cols=columns_test,
                                ID_client=identifiant)
    list_id = list(neighbours.iloc[0])

    my_df = train[train['Id_client'].isin(list_id)]
    df = pd.DataFrame()
    df = my_df[test.columns]
    df['Prêt remboursé'] = my_df['Prêt remboursé']
    df['Prêt remboursé'] = df['Prêt remboursé'].replace((0, 1), ('oui', 'non'))
    st.dataframe(df.set_index('Id_client'))

    import seaborn as sns

    target = 'Prêt remboursé'
    col = st.selectbox(label="Plus d'infos", options=test.drop(columns=['Id_client']).columns)
    # col = 'Age'

    fig = plt.figure(figsize=(10, 3))
    sns.kdeplot(train.loc[train[target] == 0, col], label=f'{target}: oui', color='green', shade=True)
    sns.kdeplot(train.loc[train[target] == 1, col], label=f'{target}: non', color='blue', shade=True)
    plt.legend()
    st.pyplot(fig)
