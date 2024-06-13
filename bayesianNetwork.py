import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.inference import VariableElimination
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder


#Bilancia il dataset con SMOTE per generare nuovi campioni con RESULTS=0 e restituire un dataset bilanciato
def balance_dataset(data):
    #Bilancia il dataset con SMOTE
    smote = SMOTE(sampling_strategy='minority')
    X = data.drop('RESULTS', axis=1)
    y = data['RESULTS']
    X_resampled, y_resampled = smote.fit_resample(X, y)
    data_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return data_resampled

def create_Bayesian_Network():
# Carica il dataset
    data = pd.read_csv('Working_Dataset.csv')

    # Seleziona un sottoinsieme di colonne
    columns_of_interest = ['RESULTS', 'IS_HIGH_CRIME_AREA', 'IS_LOW_HEALTH_AREA', 'IS_HIGH_BELOW_POVERTY_LEVEL', 'IS_LOW_PER_CAPITA_INCOME', 'IS_HIGH_UNEMPLOYMENT_RATE']
    data_subset = data[columns_of_interest]
    #lambda per mappare i result 2 in 1
    #data_subset['RESULTS'] = data_subset['RESULTS'].apply(lambda x: 1 if x == 2 else x)
    data_subset = balance_dataset(data_subset)

    # Stampa il numero di RESULTS per ogni valore
    print("Numero di campioni per ogni valore di RESULTS:")
    print(data_subset['RESULTS'].value_counts())


    # Apprendimento della struttura usando HillClimbSearch
    print("Inizio HILL")
    hc = HillClimbSearch(data_subset)
    best_model = hc.estimate(scoring_method=BicScore(data_subset))
    print("Fine HILL")

    # Apprendimento dei parametri
    model = BayesianNetwork(best_model.edges())
    model.fit(data_subset, estimator=MaximumLikelihoodEstimator)
    visualize_BN(data_subset, best_model)

    # Verifica la distribuzione condizionale appresa per una variabile
    for cpd in model.get_cpds():
        print(f"CPD per variabile {cpd.variable}")
        print(cpd)

    return model

def visualize_BN(data_subset, best_model):

    # Crea il grafo utilizzando networkx
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(data_subset.columns)
    nx_graph.add_edges_from(best_model.edges())

    # Disegna il grafo utilizzando matplotlib
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(nx_graph)  # Layout per il posizionamento dei nodi
    nx.draw(nx_graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_color='black', edge_color='gray', arrows=True)
    plt.title('Rete Bayesiana Appresa')
    plt.show()
    #salva il grafo in un file
    plt.savefig('bayesian_network.png')

# Esegui delle query sulla rete bayesiana

def query_BN(model):
    # Crea l'oggetto per l'inferenza
    inference = VariableElimination(model)

    # Query senza evidenza
    result = inference.query(variables=['RESULTS'])
    print("Risultato query senza evidenza:")
    print(result)

    # Query con evidenza
    evidence = {'IS_HIGH_CRIME_AREA': 1, 'IS_LOW_HEALTH_AREA': 1}
    result_with_evidence = inference.query(variables=['RESULTS'], evidence=evidence)
    print("Risultato query con evidenza:")
    print(result_with_evidence)

#funzione che effettua forward sampling per generare campioni casuali
def forward_sampling(model, n_samples=10):
    samples = model.simulate(n_samples=n_samples)
    print(f"{n_samples} campioni casuali generati:")
    print(samples)
    return samples

#funzione che calcola la probabilità a posteriori di una variabile data l'evidenza
def posterior_probability(model, variable, evidence):
    inference = VariableElimination(model)
    result = inference.query(variables=[variable], evidence=evidence)
    print(f"Probabilità a posteriori di {variable} data l'evidenza: \n{evidence}")
    print(result)
    result = inference.map_query(variables=[variable], evidence=evidence)
    print(result)

model = create_Bayesian_Network()
samples = forward_sampling(model, n_samples=1000)
posterior_probability(model, 'RESULTS', samples.drop(columns=['RESULTS']).iloc[0])
posterior_probability(model, 'RESULTS', samples.drop(columns=['RESULTS']).iloc[1])