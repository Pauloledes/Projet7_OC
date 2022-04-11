import pandas as pd
import numpy as np
from lime import lime_tabular
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns


def get_scores(test_labels, preds):
    print('Accuracy: %.3f' % accuracy_score(test_labels.values, preds))
    print('Precision: %.3f' % precision_score(test_labels.values, preds))
    print('Recall: %.3f' % recall_score(test_labels.values, preds))
    print('F1: %.3f' % f1_score(test_labels.values, preds))
    print('FB: %.3f' % fbeta_score(test_labels.values, preds, beta=3), '\n')


def under_sampling(df, target_name:str):
    X_train = df.drop(columns=target_name)
    y_train = df[target_name]

    rus = RandomUnderSampler(random_state=0)

    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    df_resampled = X_resampled.copy()
    df_resampled[target_name] = y_resampled

    return df_resampled


def label_encoder(df):
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0
    encoded_df = df.copy()

    # Iterate through the columns
    for col in df:
        if encoded_df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(encoded_df[col].unique())) <= 2:
                # Train on the training data
                le.fit(encoded_df[col])
                # Transform both training and testing data
                encoded_df[col] = le.transform(encoded_df[col])

                # Keep track of how many columns were label encoded
                le_count += 1
    print('%d columns were label encoded.' % le_count)
    return encoded_df


def one_hot_encoder(df):
    encoded_df = df.copy()
    encoded_df = pd.get_dummies(df)

    print('New df shape :', encoded_df.shape)

    return encoded_df


def missing_values(df):
    missing_values = df.isnull().sum()
    # compute the percentage
    missing_values_percent = 100 * df.isnull().sum() / len(df)

    # concatenate the two table
    missing_values_table = pd.concat([missing_values, missing_values_percent], axis=1)
    # give the columns more meaningful names
    mis_val_table_ren_columns = missing_values_table.rename(
        columns={0: 'Number of Missing Values', 1: 'Percentage of Entries Missing'})
    # Sort the table in descending order to see biggest values first
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0
                                                          ].sort_values('Percentage of Entries Missing',
                                                                        ascending=False).round(1)

    # Return the created missing values dataframe
    return mis_val_table_ren_columns


def align_dfs(df1, df2):
    # Align the training and testing data, keep only columns present in both dataframes
    df1, df2 = df1.align(df2, join='inner', axis=1)

    return df1, df2


def LIME_test(model, features, target, features_names, idx):
    train_features, test_features, train_labels, test_labels = train_test_split(features, target, test_size=0.2,
                                                                                random_state=42)

    model.fit(train_features, train_labels)

    print("Test R^2 Score  : ", model.score(test_features, test_labels))
    print("Train R^2 Score : ", model.score(train_features, train_labels))

    explainer = lime_tabular.LimeTabularExplainer(train_features, feature_names=features_names)

    print("Prediction : ", model.predict(test_features[idx].reshape(1, -1)))
    print("Actual :     ", list(test_labels)[idx])

    explanation = explainer.explain_instance(test_features[idx], model.predict_proba, num_features=len(features))

    return explanation.show_in_notebook()


# Defining a function to plot KDE plots
def plot_kde(df, col, target, reverse_scale=False):
    plt.figure(figsize=(12, 6))

    if reverse_scale:
        r = -1
    else:
        r = 1

    # KDE of paid loans (target == 0)
    sns.kdeplot(df.loc[df[target] == 0, col] * r, label=f'{target}: 0', color='green', shade=True)

    # KDE of defaults (target == 1)
    sns.kdeplot(df.loc[df[target] == 1, col] * r, label=f'{target}: 1', color='purple', shade=True)

    plt.xlabel('{}'.format(col))
    plt.ylabel('KDE')
    plt.title('KDE for column {}'.format(col))
    plt.show()
    plt.close()


def prepare_test(X_train, X_test, do_anom=False):
    # Instantiate a label encoder
    label_encode = LabelEncoder()
    # Iterate over columns
    for col in X_train:
        if X_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(X_train[col].unique())) <= 2:
                label_encode.fit(X_train[col])
                # apply the transformation to both train and test sets
                X_train[col] = label_encode.transform(X_train[col])
                X_test[col] = label_encode.transform(X_test[col])

    # one-hot encode the multiclass categoricals
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train_labels = X_train['TARGET']

    # align the training and testing data, keep only columns present in both dataframes
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)

    # add the target back in
    X_train['TARGET'] = X_train_labels

    if do_anom:
        X_train['DAYS_EMPLOYED_ANOM'] = X_train["DAYS_EMPLOYED"] == 365243
        X_test['DAYS_EMPLOYED_ANOM'] = X_test["DAYS_EMPLOYED"] == 365243

        # Replace the anomalous values with nan
        X_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
        X_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

        X_train.drop(columns='DAYS_EMPLOYED_ANOM', inplace=True)
        X_test.drop(columns='DAYS_EMPLOYED_ANOM', inplace=True)

    return X_train, X_test


def reduced_var_imputer(X_train, X_test):
    correlations = X_train.corr()['TARGET'].sort_values()
    corr_pos = list(correlations.sort_values(ascending=False).head(16).index)
    corr_neg = list(correlations.sort_values(ascending=True).head(16).index)
    corr = corr_pos + corr_neg

    df= X_train[corr]
    # create X_train, y_train
    X_train = df.drop('TARGET', axis=1)
    X_test = X_test[corr[1:]]
    # Feature names
    features_list = list(X_train.columns)
    X_test = X_test[corr[1:]]
    imputer = SimpleImputer(strategy='median')

    # Scale all values in 0-1 range
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit the imputer on training set
    imputer.fit(X_train)

    # transform both training and testing data
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    # fit scaler and transform
    scaler.fit(X_train)
    train = scaler.transform(X_train)
    test = scaler.transform(X_test)

    return train, test
