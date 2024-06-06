import pandas as pd

df = pd.read_csv("Food_Inspections_and_Health_Statistics_with_Date.csv")
#filtra per data
#df = df[df['Inspection Date'] >= '2023-01-01']
print(df.count())
