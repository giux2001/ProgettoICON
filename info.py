import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def balance_dataset(X, y):
    # Calcola il numero di esempi in ciascuna classe
    count_class_majority = y.value_counts()[1]  # Classe maggioritaria (pass)
    count_class_minority = y.value_counts()[0]  # Classe minoritaria (fail)

    desiderd_minority = 0.3 * (count_class_majority + count_class_minority)
    
    sampling_strategy = desiderd_minority / count_class_minority

    # Bilancia il dataset con SMOTE utilizzando la strategia calcolata
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled


# Caricamento del dataset (sostituire con il percorso corretto del tuo dataset)
df = pd.read_csv('Working_Dataset.csv')
df = df[df['NUM_INSP_AREA'] > 1]
#droppa le colonne dove results è 2
df = df[df['RESULTS'] != 2]

# Preparazione dei dati
# Esempio: selezione delle colonne di input (X) e della variabile target (y)
X = df.drop(['INSPECTION_ID', 'DBA NAME', 'FACILITY_TYPE', 'RESULTS','COMMUNITY_AREA'], axis=1)  # Escludi colonne non necessarie
y = df['RESULTS']

# Codifica delle variabili categoriche se necessario (ad esempio usando pd.get_dummies)

# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Suddivisione dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, y_train = balance_dataset(X_train, y_train)

# Creazione e addestramento del MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.001,
                    solver='adam', random_state=42, activation='relu',
                    learning_rate_init=0.01)

mlp.fit(X_train, y_train)

# Valutazione del modello
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{report}')
