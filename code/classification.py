import pandas as pd
#import pyarrow.parquet as pq
#from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet('data/traffic_data_transformed.parquet')

X = df.drop('target', axis=1)  
y = df['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')
with open('accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')


# Predici le etichette sul set di test
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizza e salva la matrice di confusione come immagine
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')  # Salva l'immagine
plt.show()