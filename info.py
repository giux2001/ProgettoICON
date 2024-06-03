import pandas as pd

df = pd.read_csv("Public_Health_Statistics_preprocessed.csv")
#permetti la visualizzazione su terminale di tutte le colonne
pd.set_option('display.max_columns', None)
#droppa le colonne sulla gonorrhea e sul childhood lead poisoning e childood blood lead level
df = df.drop(columns=['Gonorrhea in Males','Gonorrhea in Females', 'Childhood Blood Lead Level Screening', 'Childhood Lead Poisoning'])
# Calculate health index using all features except community area name, capita income, unemployment, below poverty level, and crowded housing for each community area
df['Health Index'] = df.drop(columns=['Community Area Name','Assault (Homicide)','Firearm-related','Per Capita Income', 'Unemployment', 'Below Poverty Level', 'Crowded Housing']).mean(axis=1)
pd.set_option('display.max_row', None)
#Più alto è il valore meglio è
df['Unemployment'] = 100 - df['Unemployment']
df['Below Poverty Level'] = 100 - df['Below Poverty Level']
df['Crowded Housing'] = 100 - df['Crowded Housing']

# Normalize Economic Index
df['Economic Index'] = df[['Per Capita Income', 'Below Poverty Level', 'Crowded Housing', 'Unemployment']].sum(axis=1)
df['Health Index'] = 100 - df['Health Index']

df['Criminality Index'] = df[['Assault (Homicide)', 'Firearm-related']].mean(axis=1)
# Normalizza criminality index tra 0 e 100
#df['Criminality Index'] = (df['Criminality Index'] - df['Criminality Index'].min()) / (df['Criminality Index'].max() - df['Criminality Index'].min()) * 100
#Normalizza Economic index tra 0 e 100
#df['Economic Index'] = (df['Economic Index'] - df['Economic Index'].min()) / (df['Economic Index'].max() - df['Economic Index'].min()) * 100
# Normalizza health index tra 0 e 100
#df['Health Index'] = (df['Health Index'] - df['Health Index'].min()) / (df['Health Index'].max() - df['Health Index'].min()) * 100
#print(df[['Community Area Name', 'Economic Index', 'Health Index', 'Criminality Index']])
#stampa massimo e minimo di ogni indice
df['Economic Index'] = (df['Economic Index'] / 100000) * 100
print(df[['Community Area Name', 'Economic Index', 'Health Index', 'Criminality Index']])
