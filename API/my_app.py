import json
import pickle
import sys
from API.data import DataCollection, get_kneighbors, train_nn, get_data
import numpy as np
import pandas as pd
from fastapi import FastAPI
import uvicorn
from pandas.errors import IntCastingNaNError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
sys.path.append('../')
from my_functions import functions


app = FastAPI()


def get_best_model(filename):
    with open(filename, 'rb') as file:
        my_best_model = pickle.load(file)
    return my_best_model


class Datapi(DataCollection):

    def show_my_id(self, identifiant):
        return self.predictions.loc[self.predictions['ID'] == identifiant]

    def show_my_client(self, identifiant):
        return self.overview_test.loc[self.overview_test['Id_client'] == identifiant]


my_data_collecton = Datapi()


##############################################################
# HOME #
@app.get("/")
def index():
    message = 'Welcome to my personal API which allows to read the client databases as well ' \
              'as manipulate prediction models for refundings. Have fun !'
    return message


# ##############################################################
# OLD CLIENTS #
# RAW #
@app.get("/old_clients/raw")
def get_train():
    return my_data_collecton.show_train().to_json()


# OVERVIEW #
@app.get("/old_clients/overview")
def get_train():
    return my_data_collecton.show_overview_train().to_json()


##############################################################
# NEW CLIENTS #
# RAW #
@app.get("/new_clients/raw")
def get_test():
    return my_data_collecton.show_test().to_json()


# OVERVIEW #
@app.get("/new_clients/overview")
def get_test():
    return my_data_collecton.show_overview_test().to_json()


# CLIENT ID #
@app.get("/new_clients/overview/{identifiant}")
async def overview_id(identifiant: int):
    return my_data_collecton.show_my_client(identifiant=identifiant).to_json()


# KNN #
@app.get("/new_clients/overview/{identifiant}/nn")
async def nearest_neighbours(identifiant: int):
    n_neighbours = 50
    columns_test = ['SK_ID_CURR', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'AMT_ANNUITY', 'AMT_CREDIT']

    my_nn = get_best_model('models/nn_model.pkl')
    std = get_best_model('models/standard_scaler.pkl')
    df_nn = get_data('csv_files/df_nn_light_original_train.csv')

    neighbours = get_kneighbors(df_test=my_data_collecton.original_test,
                                df_train=df_nn,
                                trained_model=my_nn,
                                cols=columns_test,
                                ID_client=identifiant,
                                standard=std,
                                n_neighbors=n_neighbours)
    list_id = list(neighbours.iloc[0])

    my_df = my_data_collecton.overview_train[my_data_collecton.overview_train['Id_client'].isin(list_id)]
    df_show = my_df[my_data_collecton.overview_test.columns]
    df_show['Prêt remboursé'] = my_df['Prêt remboursé']
    df = df_show.copy(deep=True)

    try:
        # Conversion pour meilleure visualisation
        df_show['Prêt remboursé'] = df_show['Prêt remboursé'].replace((0, 1), ('Oui', 'Non'))
        df_show["Années d'emploi"] = df_show["Années d'emploi"].astype(int)
        df_show["Annuité"] = df_show["Annuité"].astype(int)
        df_show["Crédit demandé"] = df_show["Crédit demandé"].astype(int)

    except IntCastingNaNError:
        pass

    my_list = ["Age", "Années d'emploi", "Ancienneté banque", "Annuité", "Crédit demandé"]

    df_0 = df[df['Prêt remboursé'] == 'Oui'][my_list].mean()
    df_1 = df[df['Prêt remboursé'] == 'Non'][my_list].mean()

    ranges = [(min(df[i]), max(df[i])) for i in my_list]

    df_radar = my_data_collecton.show_my_client(identifiant).drop(columns=['Id_client', 'Durée d\'endettement']).mean()

    df_all = pd.concat([df_0, df_1], axis=1)
    df_all = pd.concat([df_all, df_radar], axis=1, ignore_index=True)
    df = df_all.T

    variables = tuple(df.columns)
    client_0 = df.iloc[0]
    client_1 = df.iloc[1]
    our_client = df.iloc[2]
    client_0.dropna(inplace=True)
    client_1.dropna(inplace=True)
    our_client.dropna(inplace=True)

    my_dictionnary = {'df_show': df_show.to_json(),
                      'ranges': ranges,
                      'df': df.to_json()}

    return my_dictionnary
    # return JSONResponse(jsonable_encoder(my_dictionnary))
    # return pd.DataFrame.from_dict(my_dictionnary).to_json()
    # return json.dumps(my_dictionnary)


##############################################################
# PREDICTIONS #
@app.get("/predictions")
def get_pred():
    return my_data_collecton.show_pred().to_json()


# CLIENT ID #
@app.get("/predictions/{identifiant}")
async def get_predict_id(identifiant: int):
    df_id = my_data_collecton.show_my_id(identifiant)
    # value = np.round(df_id['Prediction'].iloc[0], 2)
    # percent = int(value * 100)
    return df_id.to_json()


# Improve score #
@app.get("/predictions/{identifiant}/improve_score")
async def improve_score(identifiant: int, amt_cred=150000, amt_dur=30):
    df_client = my_data_collecton.show_my_client(identifiant)
    my_min = int(min(my_data_collecton.original_test['AMT_CREDIT']))
    my_value = int(df_client['Crédit demandé'].iloc[0])
    max_dur = int(max(my_data_collecton.overview_train['Durée d\'endettement']))
    value_dur = int(df_client['Durée d\'endettement'].iloc[0])

    modified_test = my_data_collecton.original_test.copy()
    modified_train = my_data_collecton.original_train.copy()

    modified_test.loc[modified_test['SK_ID_CURR'] == identifiant, 'AMT_CREDIT'] = float(amt_cred)

    amt_dur = amt_cred / amt_dur
    modified_test.loc[modified_test['SK_ID_CURR'] == identifiant, 'AMT_ANNUITY'] = float(amt_dur)

    modified_train, modified_test = functions.prepare_test(modified_train, modified_test, do_anom=True)
    modified_train, modified_test = functions.reduced_var_imputer(modified_train, modified_test)

    best_model = get_best_model('../models/LGBM_model.pkl')

    predictions = best_model.predict_proba(modified_test)
    proba_remb = [i[0] for i in predictions]

    df_pred = pd.DataFrame()
    df_pred['ID'] = my_data_collecton.original_test['SK_ID_CURR']
    df_pred['Prediction'] = proba_remb

    new_proba = df_pred[df_pred['ID'] == identifiant]['Prediction'].iloc[0]
    new_df_id = df_pred.loc[df_pred['ID'] == identifiant]

    new_value = np.round(new_df_id['Prediction'].iloc[0], 3)

    percent2 = new_value * 100

    return {'new_proba': new_proba,
            'new_df_id': new_df_id.to_html(),
            'new_value': new_value,
            'percent2': percent2}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
    # print(my_data_collecton.show_pred())
