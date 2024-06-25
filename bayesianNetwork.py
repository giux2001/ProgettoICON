import pandas as pd
import pickle
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.inference import VariableElimination

def create_Bayesian_Network():
# Carica il dataset
    data = pd.read_csv('Working_Dataset.csv')

    # Seleziona un sottoinsieme di colonne
    columns_of_interest = ['RESULTS', 'IS_HIGH_CRIME_AREA', 'IS_LOW_HEALTH_AREA', 'IS_HIGH_BELOW_POVERTY_LEVEL', 'IS_LOW_PER_CAPITA_INCOME', 'IS_HIGH_UNEMPLOYMENT_RATE','NO_VIOLATIONS','VIOLATIONS_ON_MANAGEMENT_AND_SUPERVISION','VIOLATIONS_ON_HYGIENE_AND_FOOD_SECURITY','VIOLATIONS_ON_TEMPERATURE_AND_SPECIAL_PROCEDURES','VIOLATIONS_ON_FOOD_SAFETY_AND_QUALITY','VIOLATIONS_ON_INSTRUMENT_STORAGE_AND_MAINTENANCE','VIOLATIONS_ON_FACILITIES_AND_REGULATIONS','HAS_INSP_SERIOUS_VIOL']
    data_subset = data[columns_of_interest]
    data_subset['RESULTS'] = data_subset['RESULTS'].apply(lambda x: 'NOT PASS' if x == 0 else 'PASS' if x == 1 else 'PASS WITH CONDITIONS')

    # Apprendimento della struttura usando HillClimbSearch
    print("Inizio HILL")
    hc = HillClimbSearch(data_subset)
    best_model = hc.estimate(scoring_method=BicScore(data_subset))
    print("Fine HILL")

    # Apprendimento dei parametri
    model = BayesianNetwork(best_model.edges())
    model.fit(data_subset, estimator=MaximumLikelihoodEstimator)
    visualizeBayesianNetwork(model)

    # Verifica la distribuzione condizionale appresa per una variabile
    for cpd in model.get_cpds():
        print(f"CPD per variabile {cpd.variable}")
        print(cpd)

    save_BN(model)

    return model

def visualizeBayesianNetwork(bayesianNetwork):
    # Create a directed graph from the Bayesian Network edges
    G = nx.MultiDiGraph(bayesianNetwork.edges())
    
    # Define the layout of the graph nodes
    pos = nx.spring_layout(G, iterations=100, k=2, threshold=5, pos=nx.spiral_layout(G))
    
    # Draw the nodes with specified size and color
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="#00b4d8", edgecolors="#023e8a", linewidths=1.5)
    
    # Draw the labels for the nodes
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        bbox=dict(facecolor='white', edgecolor='none', pad=0.5),  # Add a white background to labels for better readability
        horizontalalignment="center",
        verticalalignment="center",
    )
    
    # Draw the edges with arrows and specific style
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=10,
        arrowstyle="-|>",
        edge_color="#0077b6",
        connectionstyle="arc3,rad=0.2",  # Use arc style for edges
        min_source_margin=1.5,
        min_target_margin=1.5,
    )
    
    # Add a title to the plot
    plt.title("Bayesian Network Graph", fontsize=15, fontweight='bold', color="#03045e")
    
    # Display the graph
    plt.show()
    
    # Clear the plot to avoid overlaps in subsequent calls
    plt.clf()

# Esegui delle query sulla rete bayesiana

def query_BN(model):
    # Crea l'oggetto per l'inferenza
    inference = VariableElimination(model)

    result = inference.query(variables=['IS_HIGH_BELOW_POVERTY_LEVEL'], evidence={'RESULTS': 'PASS'})
    print("Risultato della query che calcola la probabilità che un'area abbia un alto tasso di povertà dato che una struttura ha passato l'ispezione:")
    print(result)

    #Query che calcola la probabilità che un'area abbia un basso indice di salute dato che un ristorante ha passato l'ispezione
    result = inference.query(variables=['IS_LOW_HEALTH_AREA'], evidence={'RESULTS': 'PASS'})
    print("Risultato della query che calcola la probabilità che un'area abbia un basso indice di salute dato che una struttura ha passato l'ispezione:")
    print(result)

    #Query che calcola la probabilità che un'area abbia un alto tasso di disoccupazione dato che un ristorante ha passato l'ispezione
    result = inference.query(variables=['IS_HIGH_UNEMPLOYMENT_RATE'], evidence={'RESULTS': 'PASS'})
    print("Risultato della query che calcola la probabilità che un'area abbia un alto tasso di disoccupazione dato che una struttura ha passato l'ispezione:")
    print(result)

    #Query che calcola la probabilità che un'area abbia un alto tasso di criminalità dato che un ristorante ha passato l'ispezione
    result = inference.query(variables=['IS_HIGH_CRIME_AREA'], evidence={'RESULTS': 'PASS'})
    print("Risultato della query che calcola la probabilità che un'area abbia un alto tasso di criminalità dato che una struttura ha passato l'ispezione:")
    print(result)

    #Query che calcola la probabilità che un ristorante abbia un basso reddito pro capite dato che ha passato l'ispezione
    result = inference.query(variables=['IS_LOW_PER_CAPITA_INCOME'], evidence={'RESULTS': 'PASS'})
    print("Risultato della query che calcola la probabilità che una struttura abbia un basso reddito pro capite dato che ha passato l'ispezione:")
    print(result)

    #Query che calcola la probabilità che un ristorante abbia violazioni serie dato che ha passato l'ispezione
    result = inference.query(variables=['HAS_INSP_SERIOUS_VIOL'], evidence={'RESULTS': 'PASS'})
    print("Risultato della query che calcola la probabilità che una struttura abbia violazioni serie dato che ha passato l'ispezione:")
    print(result)

    #Query che calcola la probabilità che un ristorante abbia violazioni serie dato che ha passato l'ispezioneù
    result = inference.query(variables=['RESULTS'], evidence={'HAS_INSP_SERIOUS_VIOL': 1})
    print("Risultato della query che calcola la probabilità dei risultato delle ispezioni pdato che una struttura ha violazioni serie:")
    print(result)


#funzione che effettua forward sampling per generare campioni casuali
def forward_sampling(model, n_samples=10):
    samples = model.simulate(n_samples=n_samples).drop(columns=['RESULTS'])
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

#salva la rete bayesiana in un file
def save_BN(model):
    with open('bayesian_network.pkl', 'wb') as f:
        pickle.dump(model, f)

#carica la rete bayesiana da un file
def load_BN():
    with open('bayesian_network.pkl', 'rb') as f:
        model = pickle.load(f)
    #visualize_BN(model)
    visualizeBayesianNetwork(model)
    return model

def main():
    #model = create_Bayesian_Network()
    model = load_BN()
    samples = forward_sampling(model, n_samples=1000)
    query_BN(model)
    #cicla sul numero di campioni generati
    for i in range(0, 10):
        evidence = samples.iloc[i].to_dict()
        posterior_probability(model, 'RESULTS', evidence)

main()

