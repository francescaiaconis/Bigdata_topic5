import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import re
import sys
sys.path.append("../data_provenance_for_data_science/")
from prov_acquisition.prov_libraries.tracker import ProvenanceTracker


def is_ip_address(value):
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    return bool(ip_pattern.match(value))

def modify_host_column(dataset):
    dataset['host'] = dataset['host'].apply(lambda x: np.nan if pd.isna(x) else (1 if is_ip_address(str(x)) else 0))
    return dataset

def modify_host_column_ridotta(dataset):
    dataset[0]['host'] =dataset[0]['host'].apply(lambda x: np.nan if pd.isna(x) else (1 if is_ip_address(str(x)) else 0))
    return dataset

def val_nulli(df, df_without_missing_values, numeric_columns):
    for col in numeric_columns:
        df_with_missing_values = df[df[col].isnull()]
        if df_with_missing_values.empty:
            continue

        y_train = df_without_missing_values[col]
        X_train = df_without_missing_values.drop(col, axis=1)
        X_test = df_with_missing_values.drop(col, axis=1)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
        df.loc[df[col].isnull(), col] = y_pred_binary
    return df



numeric_columns = ['request_body_len', 'trans_depth', 'response_body_len','host']
categorical_columns = ['dest_port', 'method', 'version', 'status_code', 'response_content_type', 'request_content_type', 'target']

df = pd.read_csv('data/dataset_etichettato.csv')
df_numeric=df[numeric_columns]

df_ridotto = df_numeric.head(100)

# Create provenance tracker
tracker = ProvenanceTracker(save_on_neo4j=True)

df_ridotto= tracker.subscribe([df_ridotto])

df_numeric = modify_host_column(df_numeric)
df_ridotto = modify_host_column_ridotta(df_ridotto)

tracker.dataframe_tracking = False

df_without_missing_values = df_numeric.dropna()

df_numeric = val_nulli(df_numeric, df_without_missing_values, numeric_columns)

dataset = pd.concat([df_numeric, df[categorical_columns]], axis=1)

dataset.to_csv("data/dataset_notnull.csv", index=False)

