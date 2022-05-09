import asyncio
import json
import pickle

import aiohttp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair
import streamlit as st
import pandas as pd
from pandas.errors import IntCastingNaNError
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
from my_functions import functions
from PIL import Image
from streamlit_echarts import st_echarts
import os

os.chdir('/home/pledes/Bureau/P7/')
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])


def _scale_data(data, range):
    """scales data[1:] to ranges[0],
        inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data, range):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = range[0]
    d = data[0]

    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1

    sdata = [d]

    for d, (y1, y2) in zip(data[1:], range[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1

        sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)

    return sdata


class ComplexRadar:
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, (360. / len(variables)))

        axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True,
                             label="axes{}".format(i))
                for i in range(len(variables))]

        axes[0].set_thetagrids(angles, labels=[])

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x, 2))
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1]  # hack to invert grid
                # gridlabels aren't reversed
            gridlabel[0] = ""  # clean up origin
            ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
            # ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])

        ticks = angles
        ax.set_xticks(np.deg2rad(ticks))  # crée les axes suivant les angles, en radians
        ticklabels = variables
        ax.set_xticklabels(ticklabels, fontsize=10)  # définit les labels

        angles1 = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
        angles1[np.cos(angles1) < 0] = angles1[np.cos(angles1) < 0] + np.pi
        angles1 = np.rad2deg(angles1)
        labels = []
        for label, angle in zip(ax.get_xticklabels(), angles1):
            x, y = label.get_position()
            lab = ax.text(x, y - 0.3, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va())
            lab.set_fontsize(16)
            lab.set_fontweight('bold')
            labels.append(lab)
        ax.set_xticklabels([])

        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


def pos_client(prediction, id):
    fig = plt.figure()
    ax = prediction['Prediction'].hist(grid=False)

    for bar in ax.containers[0]:
        # get x midpoint of bar
        x = bar.get_x() + 0.5 * bar.get_width()

        # set bar color based on x
        if x < 0.4:
            bar.set_color('red')
        elif 0.4 <= x <= 0.6:
            bar.set_color('yellow')
        else:
            bar.set_color('green')

    return fig


def set_echarts(value):
    option = {"series": [
        {
            "name": "Probabilités de remboursement",
            "type": 'gauge',
            'axisLine': {
                'lineStyle': {
                    'width': 10,
                    'color': [
                        [0.39, 'red'],
                        [0.6, 'yellow'],
                        [1, 'green']
                    ]
                }
            },
            'pointer': {
                'itemStyle': {
                    'color': 'auto'
                }
            },
            'axisTick': {
                'distance': -10,
                'length': 0,
                'lineStyle': {
                    'color': '#fff',
                    'width': 4
                }
            },
            'splitLine': {
                'distance': -70,
                'length': 0,
                'lineStyle': {
                    'color': '#fff',
                    'width': 4
                }
            },
            'axisLabel': {
                'color': 'auto',
                'distance': 40,
                'fontSize': 20
            },
            'detail': {
                'valueAnimation': True,
                'formatter': f'{value} %',
                'color': 'auto'
            },
            'data': [
                {
                    'value': value,
                    'name': "Score"
                }
            ]
        }
    ]
    }
    return option


async def my_new_clients(visu):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://localhost:8000/new_clients/{visu}') as resp:
            text = await resp.json()
            final_df = pd.DataFrame.from_dict(json.loads(text))
            return final_df


async def my_old_clients(visu):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://localhost:8000/old_clients/{visu}') as resp:
            text = await resp.json()
            final_df = pd.DataFrame.from_dict(json.loads(text))
            return final_df


async def my_df_client(id_client):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://localhost:8000/new_clients/overview/{id_client}') as resp:
            text = await resp.json()
            final_df = pd.DataFrame.from_dict(json.loads(text))
            return final_df


async def my_predictions(id_client):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://localhost:8000/predictions/{id_client}') as resp:
            text = await resp.json()
            final_df = pd.DataFrame.from_dict(json.loads(text))
            return final_df


async def n_neighbours_client(id_client):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://localhost:8000/new_clients/overview/{id_client}/nn') as resp:
            text1 = await resp.json()
            final = json.loads(text1)
            return final


@st.cache(allow_output_mutation=True)
def get_data(filename):
    data = pd.read_csv(filename)

    try:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    except KeyError:
        pass

    return data


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

with header:
    st.title("Dashboard interactif")

    col1, col2 = st.columns([10, 1])

    col1.text("Ce dashboard a pour but d'aider les chargés de relation client afin qu'ils puissent à la fois "
              "expliquer de façon la plus transparente possible les décisions d’octroi de crédit, "
              "mais également permettre à leurs\n"
              "clients de disposer de leurs informations personnelles et de les explorer facilement. \n")
    col1.text("Dans le cas où la prédiction de remboursement est inférieure à 60 %, une tentative d\'amélioration \n"
              "de ce dernier est proposée.")

    image_name = 'logo.png'
    image = Image.open(image_name)

    col2.image(image_name)

with st.spinner('Chargement des anciens clients'):
    train = asyncio.run(my_new_clients(visu='overview'))
    original_train = asyncio.run(my_old_clients(visu='raw'))


with st.spinner('Chargement des nouveaux clients'):
    test = asyncio.run(my_new_clients(visu='overview'))
    original_test = asyncio.run(my_new_clients(visu='raw'))


col1, col2 = st.columns([6, 10])

with col1:
    st.markdown("Merci d'entrer un identifiant client :")
    identifiant = st.selectbox(label='', options=test['Id_client'])

    df_id = asyncio.run(my_predictions(identifiant))
    df_client = asyncio.run(my_df_client(identifiant))

    st.text('Prédictions de remboursement')
    st.dataframe(df_id)

    value = np.round(df_id['Prediction'].iloc[0], 2)
    percent = int(value * 100)
    st.text(f"Ce client a {percent} % de chances de rembourser")

    if percent >= 50:
        st.write("Prêt accordé ! ✔")

    if percent < 50:
        st.write("Prêt refusé ! ❌")

    st.markdown(f'Position du client {identifiant} dans la base des nouveaux clients')
    st.empty()

    st_echarts(options=set_echarts(percent), width="100%")

    my_min = int(min(original_test['AMT_CREDIT']))
    my_value = int(df_client['Crédit demandé'].iloc[0])

    if value < 0.4:
        st.text("Il semblerait que ce client ne soit pas à même de rembourser. Il apparaît également difficile \n"
                "de tenter de modifier les termes de sa demande pour augmenter les chances de remboursement.")

    if value > 0.6:
        st.text("Ce client est fortement susceptible de rembourser le prêt. Nul besoin de tenter d'améliorer \n"
                "son score qui est déjà très élevé !")

    if 0.4 <= value <= 0.6:
        st.text("Les probabilités de remboursement de la part de ce client sont modérées mais il est \n"
                "possible de modifier certains de ses termes pour améliorer ses chances.  ")

with col2:
    st.title("Vue générale")
    st.text('Notre client')

    try:
        # Conversion pour meilleure visualisation
        df_client["Années d'emploi"] = df_client["Années d'emploi"].astype(int)
        df_client["Annuité"] = df_client["Annuité"].astype(int)
        df_client["Crédit demandé"] = df_client["Crédit demandé"].astype(int)
        df_client["Durée d\'endettement"] = df_client["Durée d\'endettement"].astype(int)

    except IntCastingNaNError:
        pass

    st.dataframe(df_client.set_index('Id_client'))
    st.text("Comparaison de notre client avec les rembourseurs et non-rembourseurs")

    all_infos = asyncio.run(n_neighbours_client(identifiant))
    df = pd.DataFrame.from_dict(json.loads(all_infos['df']['0']))
    df_show = pd.DataFrame.from_dict(json.loads(all_infos['df_show']['0']))

    client_0 = df.iloc[0]
    client_1 = df.iloc[1]
    our_client = df.iloc[2]
    ranges = all_infos['ranges']
    variables = ("Age", "Années d'emploi", "Ancienneté banque", "Annuité", "Crédit demandé")

    # plotting
    fig1 = plt.figure(figsize=(6, 6))
    radar = ComplexRadar(fig1, variables, list(ranges.values()))
    radar.plot(our_client, label=f'Notre client')
    radar.fill(our_client, alpha=0.2)

    radar.plot(client_0, label='Moyenne des clients similaires ayant remboursé', color='g')
    radar.plot(client_1,
               label='Moyenne des clients similaires n\'ayant pas remboursé',
               color='r')

    fig1.legend(bbox_to_anchor=(1.7, 1))

    st.pyplot(fig1)

    with st.expander("Détails"):
        st.text("Clients similaires dans l'historique de la base de données (50 plus proches)")
        st.dataframe(df_show)  # .set_index('Id_client'))

with st.expander("Plus d'informations"):
    my_columns = test.drop(columns=['Id_client']).columns
    st.text(f'Il est ici possible de visualiser la tendance globale des anciens clients ayant remboursé '
            f'ou non selon un critère donné')
    target = 'Prêt remboursé'
    col = st.selectbox(label="", options=my_columns)
    fig = plt.figure(figsize=(10, 3))
    sns.kdeplot(train.loc[train[target] == 'Oui', col], label='oui', color='green', shade=True)
    sns.kdeplot(train.loc[train[target] == 'Non', col], label='non', color='blue', shade=True)

    lim = plt.gca().get_ylim()
    value_client = df_client[col].iloc[0]

    plt.vlines(value_client,
               lim[0],
               lim[1],
               colors='red',
               label=f'Notre client : {value_client}')

    plt.legend(title=target)
    st.pyplot(fig)

with st.expander("+"):
    df = get_data('csv_files/d_head.csv')
    col1, col2 = st.columns([2, 3])
    col1.text('En réalité le dataframe utilisé pour l\'implémentation du modèle de prédiction'
              'est plus complexe et \n ressemble à ça')
    col1.dataframe(df)

    image_name = 'notebooks/fi.jpg'
    image = Image.open(image_name)

    col2.image(image_name, caption='Importance des différentes variables dans la prédiction')

with st.expander('Comment améliorer ce score ?'):
    if percent >= 60:
        st.text('Pas de proposition ici')

    else:
        st.write('Bien que de faibles importances dans le modèle, la modification de ces variables peut entraîner une '
                 'amélioration')
        best_model = get_best_model(filename="models/LGBM_model.pkl")
        col1, col2 = st.columns([6, 10])

        with col1:
            amnt_cred = st.slider(label='Diminuer le crédit demandé',
                                  min_value=my_min,
                                  max_value=my_value,
                                  value=my_value,
                                  step=10000)

            max_dur = int(max(train['Durée d\'endettement']))
            value_dur = int(df_client['Durée d\'endettement'].iloc[0])

            amt_dur = st.slider(label='Augmenter la durée d\'emprunt',
                                min_value=value_dur,
                                max_value=max_dur,
                                value=value_dur,
                                step=5)

            modified_test = original_test.copy()
            modified_train = original_train.copy()
            # st.dataframe(modified_test)
            modified_train, modified_test = functions.prepare_test(modified_train, modified_test, do_anom=True)
            st.dataframe(modified_test.loc[modified_test['SK_ID_CURR'] == identifiant, 'AMT_CREDIT'])
            modified_test.loc[modified_test['SK_ID_CURR'] == identifiant, 'AMT_CREDIT'] = float(amnt_cred)
            st.dataframe(modified_test.loc[modified_test['SK_ID_CURR'] == identifiant, 'AMT_CREDIT'])

            amt_dur = amnt_cred / amt_dur
            modified_test.loc[modified_test['SK_ID_CURR'] == identifiant, 'AMT_ANNUITY'] = float(amt_dur)
            modified_train, modified_test = functions.reduced_var_imputer(modified_train, modified_test)

            predictions = best_model.predict_proba(modified_test)
            proba_remb = [i[0] for i in predictions]

            df_pred = pd.DataFrame()
            df_pred['ID'] = original_test['SK_ID_CURR']
            df_pred['Prediction'] = proba_remb

            new_proba = df_pred[df_pred['ID'] == identifiant]['Prediction'].iloc[0]
            new_df_id = df_pred.loc[df_pred['ID'] == identifiant]

            new_value = np.round(new_df_id['Prediction'].iloc[0], 3)

            # Make figure
            fig = pos_client(df_pred, new_df_id)
            lim = plt.gca().get_ylim()
            plt.vlines(df_id['Prediction'].iloc[0],
                       lim[0],
                       lim[1],
                       colors='blue',
                       label=f'Ancien score : {value}')

            plt.vlines(new_df_id['Prediction'].iloc[0],
                       lim[0],
                       lim[1],
                       colors='green',
                       label=f'Nouveau score : {new_value}')
            plt.legend()
            st.pyplot(fig)
            percent2 = new_value * 100
            st_echarts(options=set_echarts(percent2), width="100%")
            dif = percent2 - percent

        with col2:
            st.text('Prédictions de remboursement')
            st.dataframe(new_df_id)
            st.write(f'Nouveau score : {percent2} %')

            if dif > 0:
                st.write(f'Ce choix a permis une amélioration de {np.round(dif, 1)}% !')
            elif dif < 0:
                st.write(f'Ce choix a entraîné une diminution de {abs(np.round(dif, 1))}% ...')
            else:
                st.write('Le score n\'a pas changé.')
