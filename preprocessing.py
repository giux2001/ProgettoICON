import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import re
from pyswip import Prolog

FOOD_INSPECTIONS = "dataset/Food_Inspections_20240520.csv"
PUBLIC_HEALTH_STATISTICS = "dataset/Public_Health_Statistics_-_Selected_public_health_indicators_by_Chicago_community_area_-_Historical_20240520.csv"

def preprocesse_food_inspections():

    df = pd.read_csv(FOOD_INSPECTIONS)

    colonne_da_eliminare = ["License #", "AKA Name", "State", "Location", "Address", "City", "Zip", "Inspection Type"]
    df.drop(colonne_da_eliminare, axis=1, inplace=True)
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"]) #conversione della data in formato datetime
    df= df[df["Inspection Date"] >= "2019-01-01"]
    df = df.drop_duplicates(subset=["DBA Name"])
    
    mapping_rischi = {
        "Risk 1 (High)": 1,
        "Risk 2 (Medium)": 2,
        "Risk 3 (Low)": 3
    }

    df["Risk"] = df["Risk"].map(mapping_rischi)
    df["Violations"] = df["Violations"].apply(estrazione_codici)

    valori_da_rimuovere = ["No Entry", "Out of Business", "Business Not Located", "Not Ready"]
    df = df[~df['Results'].isin(valori_da_rimuovere)]

    mapping_results = { #Mappatura di Pass w/ Conditions e Pass in 1, Fail in 0 (potremmo anche fare Pass Conditions a altro valore)
    "Pass w/ Conditions": 2, #oppure possiamo proprio rimuovere Pass w/ Conditions
    "Fail": 0,
    "Pass": 1,
    }

    df["Results"] = df["Results"].map(mapping_results)

    df["Facility Type"] = df["Facility Type"].replace("Daycare (2 - 6 Years)", "Daycare")
    #Sostituisci daycare (under 2 years) con daycare
    df["Facility Type"] = df["Facility Type"].replace("Daycare (Under 2 Years)", "Daycare")
    #Sostituisci daycare (2 - 6 years) with daycare
    df["Facility Type"] = df["Facility Type"].replace("Daycare Above and Under 2 Years", "Daycare")
    #Sostituisci shared kitchen user (Long Term) con shared kitchen
    df["Facility Type"] = df["Facility Type"].replace("Shared Kitchen User (Long Term)", "Shared Kitchen")
    #cancella le righe con Facility Type che hanno meno di 30 occorrenze
    df = df.groupby('Facility Type').filter(lambda x: len(x) > 30)

    df = df.dropna(subset=['Latitude', 'Longitude'])

    #Associazione coordinate a community area
    coordinates = list(df[['Latitude', 'Longitude']].values)
    i = 1
    for lat, lon in coordinates:
        community = find_community_area(lat, lon)
        #assegna la community area alla colonna Location
        df.loc[(df['Latitude'] == lat) & (df['Longitude'] == lon), 'Location'] = community
        print(f" Numero Esempio: {i}, Lat: {lat}, Lon: {lon}, Community Area: {community}")
        i += 1

    df.drop(["Latitude", "Longitude"], axis=1, inplace=True)
    #Rinomina la colonna Location in Community Area
    df.rename(columns={"Location": "Community Area Name"}, inplace=True)
    #salva il dataset in un file csv
    #df.to_csv("dataset/Food_Inspections_preprocessed.csv", index=False)

    #df = pd.read_csv("dataset/Food_Inspections_preprocessed.csv")
    #visualizza tutte le occorrenze dei valori della colonna Facility Type
    #df = pd.read_csv("dataset/Food_Inspections_preprocessed.csv")
    #Sostituisci daycare (2 - 6 years) con daycare
    
    df.to_csv("dataset/Food_Inspections_preprocessed.csv", index=False)

    print("DATASET FOOD INSPECTIONS")
    return df

def preprocesse_public_health_statistics():
    df = pd.read_csv(PUBLIC_HEALTH_STATISTICS)
    df = df.drop_duplicates(subset=["Community Area Name"])
    # Eliminazione delle colonne non necessarie
    colonne_da_eliminare = ["Dependency", "No High School Diploma"]
    
    #Rimangono Community Area Name, Omicidi, Cancer, Diabete, Infant Mortality, Stroke, Poverty Level, Crowded Housing, Per Capita Income, Unemployment
    df.drop(colonne_da_eliminare, axis=1, inplace=True)
    print("DATASET PUBLIC HEALTH STATISTICS")
    df.to_csv("dataset/Public_Health_Statistics_preprocessed.csv", index=False)
    return df

def estrazione_codici(violazioni):
    if pd.isnull(violazioni):
        return []
    codes = re.findall(r'(?<=\| )(\d+)\.|^(\d+)\.|\| (\d+)\.|^\"(\d+)\.', violazioni)
    # Unisce i numeri estratti in una lista
    codes = [code[0] if code[0] != '' else code[1] if code[1] != '' else code[2] if code[2] != '' else code[3] if code[3] != '' else "" for code in codes]
    return codes

# Funzione per trovare la community area data una latitudine e longitudine
def find_community_area(lat, lon):
    community_areas = gpd.read_file("chicago-community-areas.geojson")
    point = Point(lon, lat)
    for idx, area in community_areas.iterrows():
        if area['geometry'].contains(point):
            return area['community']
    return None

def joinDataset(food, health):
    #food = pd.read_csv("dataset/Food_Inspections_preprocessed.csv")
    #health = pd.read_csv("dataset/Public_Health_Statistics_preprocessed.csv")
    #Rendi minuscole le stringhe della colonna Community Area Name
    food["Community Area Name"] = food["Community Area Name"].str.lower()
    health["Community Area Name"] = health["Community Area Name"].str.lower()
    df = pd.merge(food, health, on="Community Area Name", how="inner")
    df = df.dropna()
    df.to_csv("dataset/Food_Inspections_and_Health_Statistics.csv", index=False)

#Food = preprocesse_food_inspections()
#Health = preprocesse_public_health_statistics()
#joinDataset(Food, Health)


df = pd.read_csv("Food_Inspections_and_Health_Statistics.csv")
#df = df.drop(columns=['Gonorrhea in Males','Gonorrhea in Females', 'Childhood Blood Lead Level Screening', 'Childhood Lead Poisoning'])
#calcola l'indice di salute in un'area
#df['Health Index'] = df.drop(columns=['Community Area Name','Assault (Homicide)','Firearm-related','Per Capita Income', 'Unemployment', 'Below Poverty Level', 'Crowded Housing', 'Inspection ID', 'DBA Name', 'Facility Type','Risk','Inspection Date','Results','Violations']).mean(axis=1)
#df['Health Index'] = 100 - df['Health Index']
#df = df.drop(columns=['Birth Rate', 'General Fertility Rate', 'Low Birth Weight', 'Prenatal Care Beginning in First Trimester', 'Preterm Births', 'Teen Birth Rate', 'Breast cancer in females', 'Cancer (All Sites)', 'Colorectal Cancer', 'Diabetes-related', 'Infant Mortality Rate', 'Lung Cancer', 'Prostate Cancer in Males', 'Stroke (Cerebrovascular Disease)', 'Tuberculosis'])
#df['Crime Index'] = df[['Assault (Homicide)', 'Firearm-related']].mean(axis=1)
#df = df.drop(columns=['Assault (Homicide)', 'Firearm-related'])
#df = df.drop(columns=['Crowded Housing'])
#df.to_csv("Food_Inspections_and_Health_Statistics.csv", index=False)
#media degli indici di salute
print(df['Health Index'].mean())
#media degli indici di criminalità
print(df['Crime Index'].mean())


 
