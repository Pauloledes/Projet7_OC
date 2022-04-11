import pickle
import seaborn as sns
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

st.set_option('deprecation.showPyplotGlobalUse', False)


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
    distance, voisins = trained_model.kneighbors([client_list[my_index]], n_neighbors=50)
    voisins = voisins[0]

    voisins_table = pd.DataFrame()
    # Create df with k nearest neighbours
    for v in range(len(voisins)):
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

my_col, _ = st.columns([6, 10])

with my_col:
    st.markdown("Merci d'entrer un identifiant client :")
    identifiant = st.selectbox(label='', options=test['Id_client'])
    df_id = predictions.loc[predictions['ID'] == identifiant]
    df_client = test.loc[test['Id_client'] == identifiant]

col1, col2 = st.columns([6, 10])

with col1:
    st.text('Prédictions de remboursement')
    st.dataframe(df_id)
    # st.dataframe(original_test)
    # original_id = original_test[original_test[]]
    value = np.round(df_id['Prediction'].iloc[0], 3)
    percent = value * 100
    st.text(f"Ce client a {percent} % de chances de rembourser")

    if percent >= 50:
        st.write("Prêt accordé ! ✔")

    if percent < 50:
        st.write("Prêt refusé ! ❌")

    st.markdown(f'Position du client {identifiant} dans la base des nouveaux clients')


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


    fig = pos_client(predictions, df_id)
    lim = plt.gca().get_ylim()
    plt.vlines(df_id['Prediction'].iloc[0],
               lim[0],
               lim[1],
               colors='blue',
               label=f'Notre client \n Score : {value}')
    plt.legend()

    st.pyplot(fig)

    my_min = int(min(original_test['AMT_CREDIT']))
    my_value = int(df_client['Crédit demandé'].iloc[0])

with col2:
    st.title("Vue générale")
    neighbours = get_kneighbors(df_test=original_test, df_train=df_nn, trained_model=my_nn, cols=columns_test,
                                ID_client=identifiant)
    list_id = list(neighbours.iloc[0])

    my_df = train[train['Id_client'].isin(list_id)]
    df_show = my_df[test.columns]
    df_show['Prêt remboursé'] = my_df['Prêt remboursé']
    df = df_show.copy(deep=True)

    df_show['Prêt remboursé'] = df_show['Prêt remboursé'].replace((0, 1), ('Oui', 'Non'))
    my_list = ["Age", "Années d'emploi", "Ancienneté banque", "Annuité", "Crédit demandé"]

    df_0 = df[df['Prêt remboursé'] == 'Oui'][my_list].mean()
    df_1 = df[df['Prêt remboursé'] == 'Non'][my_list].mean()

    ranges = [(min(df[i]), max(df[i])) for i in my_list]
    client_v = [df_client[i] for i in my_list]

    df_radar = df_client.drop(columns=['Id_client', 'Durée d\'endettement']).mean()

    # df_radar = pd.concat(df, df_0)
    df_all = pd.concat([df_0, df_1], axis=1)
    df_all = pd.concat([df_all, df_radar], axis=1, ignore_index=True)
    df = df_all.T


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


    class ComplexRadar():
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


    variables = ("Age", "Années d'emploi", "Ancienneté banque", "Annuité", "Crédit demandé")
    client_0 = df.iloc[0]
    client_1 = df.iloc[1]
    our_client = df.iloc[2]
    # plotting
    fig1 = plt.figure(figsize=(6, 6))
    radar = ComplexRadar(fig1, variables, ranges)
    radar.plot(our_client, label=f'Notre client')
    radar.fill(our_client, alpha=0.2)

    radar.plot(client_0, label='Moyenne des clients similaires ayant remboursé', color='g')
    radar.plot(client_1,
               label='Moyenne des clients similaires n\'ayant pas remboursé',
               color='r')

    fig1.legend(bbox_to_anchor=(1.7, 1))

    st.pyplot(fig1)

    with st.expander("Détails"):
        st.text('Notre client')
        st.dataframe(df_client.set_index('Id_client'))
        st.text("Clients similaires dans l'historique de la base de données")
        st.dataframe(df_show.set_index('Id_client'))

with st.expander("Plus d'informations"):
    target = 'Prêt remboursé'
    col = st.selectbox(label="", options=test.drop(columns=['Id_client']).columns)
    fig = plt.figure(figsize=(10, 3))
    sns.kdeplot(train.loc[train[target] == 'Oui', col], label='oui', color='green', shade=True)
    sns.kdeplot(train.loc[train[target] == 'Non', col], label='non', color='blue', shade=True)
    plt.legend(title=target)
    st.pyplot(fig)

with st.expander('Comment améliorer ce score ?'):
    best_model = get_best_model(filename="LGBM_model.pkl")
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

        modified_test.loc[modified_test['SK_ID_CURR'] == identifiant, 'AMT_CREDIT'] = float(amnt_cred)

        amt_dur = amnt_cred / amt_dur
        modified_test.loc[modified_test['SK_ID_CURR'] == identifiant, 'AMT_ANNUITY'] = float(amt_dur)

        modified_train, modified_test = functions.prepare_test(modified_train, modified_test, do_anom=True)
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
        dif = percent2 - percent

    with col2:
        st.text('Prédictions de remboursement')
        st.dataframe(new_df_id)
        st.write(f'Nouveau score : {percent2} %')
        st.write(f'Ce choix a permis une amélioration de {np.round(dif, 1)}% !')
