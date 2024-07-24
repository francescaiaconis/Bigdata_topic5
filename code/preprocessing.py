from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, DoubleType, ArrayType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline


spark = SparkSession.builder.appName("TrafficData").getOrCreate()

file_path = 'file:///C:/Users/fraia/Desktop/prog_bigdata/code/data/dataset_notnull.csv'
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Elimina colonne non necessarie
numeric_columns = ['request_body_len', 'trans_depth', 'response_body_len', 'host']
categorical_columns = ['dest_port', 'method', 'version', 'status_code', 'response_content_type', 'request_content_type']
df = df.select(numeric_columns + categorical_columns + ['target'])

# Riduce il numero di valori diversi per ogni colonna categorica
def reduce_categories(df, categorical_columns, top_n=10):
    for col_name in categorical_columns:
        top_categories = [row[col_name] for row in df.groupBy(col_name).count().orderBy('count', ascending=False).limit(top_n).select(col_name).collect()]
        df = df.withColumn(col_name, F.when(F.col(col_name).isin(top_categories), F.col(col_name)).otherwise(F.lit('Other')))
    return df

df = reduce_categories(df, categorical_columns)

# Crea una lista di StringIndexer e OneHotEncoder per ogni colonna categorica
indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + "_index") for col_name in categorical_columns]
encoders = [OneHotEncoder(inputCols=[col_name + "_index"], outputCols=[col_name + "_one_hot"]) for col_name in categorical_columns]

# Pipeline per l'indicizzazione e la codifica one-hot
pipeline = Pipeline(stages=indexers + encoders)
model = pipeline.fit(df)
df_encoded = model.transform(df)

def separa_one_hot(df, col_name_prefix, original_col_name, top_categories):
    def extract_value(vectors, idx):
        return float(vectors[idx]) if idx < len(vectors) else 0.0

    extract_udfs = [udf(lambda vectors, idx=i: extract_value(vectors, idx), DoubleType()) for i in range(len(top_categories))]

    for i, category in enumerate(top_categories):
        df = df.withColumn(f"{original_col_name}_{category}", extract_udfs[i](col(f"{col_name_prefix}")))
    
    return df

for col_name in categorical_columns:
    top_categories = [row[col_name] for row in df.groupBy(col_name).count().orderBy('count', ascending=False).limit(10).select(col_name).collect()]
    df_encoded = separa_one_hot(df_encoded, col_name + "_one_hot", col_name, top_categories)

columns_to_drop = [col_name + "_index" for col_name in categorical_columns] + [col_name + "_one_hot" for col_name in categorical_columns] + ['dest_port', 'method', 'version', 'status_code', 'response_content_type', 'request_content_type']
df_encoded = df_encoded.drop(*columns_to_drop)

assembler = VectorAssembler(inputCols=numeric_columns, outputCol='features')
df_features = assembler.transform(df_encoded)

scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(df_features)
df_scaled = scaler_model.transform(df_features)

to_dense = F.udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
df_scaled = df_scaled.withColumn('scaled_features_dense', to_dense(F.col('scaled_features')))

for i, col_name in enumerate(numeric_columns):
    df_scaled = df_scaled.withColumn(f"{col_name}_scaled", F.col('scaled_features_dense').getItem(i))

df_scaled = df_scaled.drop('features', 'scaled_features', 'scaled_features_dense')
df_scaled = df_scaled.select([f"{col}_scaled" for col in numeric_columns])

df_encoded = df_encoded.withColumn("index", F.monotonically_increasing_id())
df_scaled = df_scaled.withColumn("index", F.monotonically_increasing_id())

df_final = df_encoded.join(df_scaled, on="index", how='inner')
df_final = df_final.drop('index')
df_final = df_final.drop(*numeric_columns)
df_final.printSchema()
df_final.show(5)

# Stampa le colonne per verifica
print(df_final.columns)
output_path_parquet = "file:///C:/Users/fraia/Desktop/prog_bigdata/code/data/traffic_data_transformed.parquet"
df_final.write.format("parquet").save(output_path_parquet, mode="overwrite")
print("File salvato in: data/traffic_data_transformed.parquet")

df_from_parquet = spark.read.parquet(output_path_parquet)
df_from_parquet.show(truncate=False)

spark.stop()
