from data import train_nn, get_data
import pickle

columns_test = ['SK_ID_CURR', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'AMT_CREDIT']
original_train = get_data('../csv_files/original_train.csv')
original_train = original_train.head(1000)
my_nn, df_nn, std = train_nn(original_train, cols=columns_test)

with open('../models/nn_model.pkl', 'wb') as file:
    pickle.dump(my_nn, file)


original_train.to_csv('../csv_files/light_original_train.csv')
df_nn.to_csv('../csv_files/df_nn_light_original_train.csv')

with open('../models/standard_scaler.pkl', 'wb') as file:
    pickle.dump(std, file)
