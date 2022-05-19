import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os

# os.chdir('/home/pledes/Bureau/P7/API/')


def get_data(filename):
    data = pd.read_csv(filename)

    try:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    except KeyError:
        pass

    return data


offline = False
csv_string = 'csv_files'

if offline:
    csv_string = '../' + csv_string


class DataCollection:
    # original_test = get_data('csv_files/original_test.csv')
    # original_train = get_data('csv_files/light_original_train.csv')
    # predictions = get_data('csv_files/submission.csv')
    # overview_test = get_data('csv_files/vue_generale_test.csv')
    # overview_train = get_data('csv_files/vue_generale_train.csv')
    original_test = get_data(f'{csv_string}/original_test.csv')
    original_train = get_data(f'{csv_string}/light_original_train.csv')
    predictions = get_data(f'{csv_string}/submission.csv')
    overview_test = get_data(f'{csv_string}/vue_generale_test.csv')
    overview_train = get_data(f'{csv_string}/vue_generale_train.csv')

    def show_test(self):
        return self.original_test

    def show_train(self, ):
        return self.original_train

    def show_overview_test(self):
        return self.overview_test

    def show_overview_train(self):
        return self.overview_train

    def show_pred(self):
        return self.predictions


def train_nn(df_train, cols, n_neighbors=4):
    # Collecting data
    df_nn = df_train[cols]
    # Collecting target
    df_nn.loc['TARGET'] = df_train['TARGET']
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


def get_kneighbors(df_test, df_train, trained_model, cols, ID_client, standard, n_neighbors=50):
    # Collecting index of requested ID_client
    my_index = df_test[cols][df_test[cols]['SK_ID_CURR'] == ID_client].index[0]

    # Standaridation of data to predict
    client_list = standard.transform(df_test[cols])

    # Prediction
    distance, voisins = trained_model.kneighbors([client_list[my_index]], n_neighbors=n_neighbors)
    voisins = voisins[0]

    voisins_table = pd.DataFrame()
    # Create df with k nearest neighbours
    for v in range(len(voisins)):
        voisins_table[v] = df_train.iloc[voisins[v]]

    return voisins_table

