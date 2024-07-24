import pandas as pd
from ydata_profiling import ProfileReport
# Carica il dataset
df = pd.read_csv('data/dataset_etichettato.csv')

# Genera il report di profiling
profile = ProfileReport(df, title="Traffic Data Profiling Report", explorative=True)

# Esporta il report come file HTML
profile.to_file("output/traffic_data_report.html")