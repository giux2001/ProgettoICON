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

def save_Food_and_Health_in_KB():
    df = pd.read_csv("dataset/Food_Inspections_and_Health_Statistics.csv")
    #Definisci fatti per il dataset Food Inspections da salvare in facts.pl
    prolog = Prolog()

    for index, row in df.iterrows():
        #Fatti per Food Inspections
        inspection_id = f"inspection_id({row['Inspection ID']})"
        inspections_facts = [f"facility_name({inspection_id}, '{row['DBA Name']}')",
                             f"facility_type({inspection_id}, '{row['Facility Type']}')",
                             f"risk({inspection_id}, {row['Risk']})",
                             f"results({inspection_id}, {row['Results']})",
                             f"violations({inspection_id}, {row['Violations']})",
                             f"community_area({inspection_id}, '{row['Community Area Name']}')",
                             f"inspection_date({inspection_id}, '{row['Inspection Date']}')"]
                             
        #Fatti per Community Area
        community_area = f"community_area('{row['Community Area Name']}')"
        community_area_facts = [f"inspection_in_community_area({inspection_id}, {community_area})",
                                f"birth_rate({community_area}, {row['Birth Rate']})",
                                f"general_fertility_rate({community_area}, {row['General Fertility Rate']})",
                                f"low_birth_weight({community_area}, {row['Low Birth Weight']})",
                                f"prenatal_care_beginning_in_first_trimester({community_area}, {row['Prenatal Care Beginning in First Trimester']})",
                                f"premterme_births({community_area}, {row['Preterm Births']})",
                                f"teen_birth_rate({community_area}, {row['Teen Birth Rate']})",
                                f"assault({community_area}, {row['Assault (Homicide)']})",
                                f"breast_cancer_in_females({community_area}, {row['Breast cancer in females']})",
                                f"cancer_all_sites({community_area}, {row['Cancer (All Sites)']})",
                                f"colorectal_cancer({community_area}, {row['Colorectal Cancer']})",
                                f"diabetes_related({community_area}, {row['Diabetes-related']})",
                                f"firearm_related({community_area}, {row['Firearm-related']})",
                                f"infant_mortality_rate({community_area}, {row['Infant Mortality Rate']})",
                                f"lung_cancer({community_area}, {row['Lung Cancer']})",
                                f"prostate_cancer_in_males({community_area}, {row['Prostate Cancer in Males']})",
                                f"stroke({community_area}, {row['Stroke (Cerebrovascular Disease)']})",
                                f"childhood_poisoning({community_area}, {row['Childhood Lead Poisoning']})",
                                f"gonorrhea_in_females({community_area}, {row['Gonorrhea in Females']})",
                                f"gonorrhea_in_males({community_area}, {row['Gonorrhea in Males']})",
                                f"tubercolosis({community_area}, {row['Tuberculosis']})",
                                f"below_poverty_level({community_area}, {row['Below Poverty Level']})",
                                f"crowded_housing({community_area}, {row['Crowded Housing']})",
                                f"per_capita_income({community_area}, {row['Per Capita Income']})",
                                f"unemployment({community_area}, {row['Unemployment']})"]




