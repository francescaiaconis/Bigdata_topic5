import pandas as pd
from ydata_profiling import ProfileReport



df = pd.read_parquet('data/traffic_data_transformed.parquet')
df=df.head(1000)
# Genera il report di profiling
profile = ProfileReport(df, title="Traffic Data Profiling Report Post", explorative=True)

# Esporta il report come file HTML
profile.to_file("output/traffic_data_report_post.html")