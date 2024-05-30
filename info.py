import pandas as pd

df = pd.read_csv("dataset/Food_Inspections_and_Health_Statistics.csv")
print(len(set(df["Community Area Name"])))