import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.inference import VariableElimination

# Carica il dataset
data = pd.read_csv('Working_Dataset.csv')

# Seleziona un sottoinsieme di colonne
columns_of_interest = ['RESULTS', 'IS_HIGH_CRIME_AREA', 'IS_LOW_HEALTH_AREA', 'IS_HIGH_BELOW_POVERTY_LEVEL', 'IS_LOW_PER_CAPITA_INCOME', 'IS_HIGH_UNEMPLOYMENT_RATE']
data_subset = data[columns_of_interest]

# Apprendimento della struttura usando HillClimbSearch
print("Inizio HILL")
hc = HillClimbSearch(data_subset)
best_model = hc.estimate(scoring_method=BicScore(data_subset))
print("Fine HILL")

# Apprendimento dei parametri
model = BayesianNetwork(best_model.edges())
model.fit(data_subset, estimator=MaximumLikelihoodEstimator)

# Verifica la distribuzione condizionale appresa per una variabile
for cpd in model.get_cpds():
    print(f"CPD per variabile {cpd.variable}")
    print(cpd)

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

# Esegui delle query sulla rete bayesiana
inference = VariableElimination(model)

# Query senza evidenza
result = inference.query(variables=['RESULTS'])
print("Risultato query senza evidenza:")
print(result)

# Query con evidenza
evidence = {'IS_HIGH_CRIME_AREA': 1, 'IS_LOW_HEALTH_AREA': 0}
result_with_evidence = inference.query(variables=['RESULTS'], evidence=evidence)
print("Risultato query con evidenza:")
print(result_with_evidence)
