from pyswip import Prolog
import pandas as pd

def save_Food_and_Health_in_KB():
    df = pd.read_csv("Food_Inspections_and_Health_Statistics.csv")
    #Togli gli apici per i DBA Name
    df['DBA Name'] = df['DBA Name'].str.replace("'", "")
    df["Facility Type"] = df["Facility Type"].str.replace("'", "")

    #Definisci fatti per il dataset Food Inspections da salvare in facts.pl
    prolog = Prolog()

    with open("facts.pl", "a") as file:
            file.write(":-style_check(-discontiguous).\n")
    # se il file è vuoto scrivi style discontiguos per evitare errori, se il file non è vuoto cancella il contenuto e scrivi style discontiguos
    #with open("facts.pl", "w") as file:
        #file.write(":-style_check(-discontiguous)\n")

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
                                f"crime_index({community_area}, {row['Crime Index']})",
                                f"health_index({community_area}, {row['Health Index']})",
                                f"below_poverty_level({community_area}, {row['Below Poverty Level']})",
                                f"per_capita_income({community_area}, {row['Per Capita Income']})",
                                f"unemployment({community_area}, {row['Unemployment']})"]
    
        
        #Salva i fatti nel file facts.pl, all'inizio del file scrivi style discontiguos per evitare errori
        with open("facts.pl", "a") as file:
            for fact in inspections_facts:
                file.write(fact + ".\n")
            for fact in community_area_facts:
                file.write(fact + ".\n")
        


def create_KB():
    
    #save_Food_and_Health_in_KB()
    prolog = Prolog()
    prolog.consult("facts.pl")
   
    #Definizione delle clausole nella Knowledge Base
    
    #Clausola per trovare il numero di ispezioni in un'area
    prolog.assertz("total_inspections_in_area(CommunityArea, Count) :-findall(InspectionID, inspection_in_community_area(InspectionID, CommunityArea), Inspections),length(Inspections, Count)")
    #print(list(prolog.query("total_inspections_in_area(community_area('albany park'), Count)")))
    #Clausola per trovare il numero di ispezioni fallite in un'area
    prolog.assertz("failed_inspections_in_area(CommunityArea, Count) :-findall(InspectionID, (inspection_in_community_area(InspectionID, CommunityArea), results(InspectionID, 0)), FailedInspections),length(FailedInspections, Count)")
    #print(list(prolog.query("failed_inspections_in_area(community_area('albany park'), Count)")))
    #Clausola per trovare la percentuale di ispezioni fallite in un'area
    prolog.assertz("percentage_failed_inspections_in_area(CommunityArea, Percentage) :-total_inspections_in_area(CommunityArea, TotalCount),failed_inspections_in_area(CommunityArea, FailedCount),TotalCount > 0, Percentage is (FailedCount / TotalCount) * 100")
    #print(list(prolog.query("percentage_failed_inspections_in_area(community_area('albany park'), Percentage)")))
    #Clausola per trovare il numero di ispezioni passate in un'area
    prolog.assertz("passed_inspections_in_area(CommunityArea, Count) :-findall(InspectionID, (inspection_in_community_area(InspectionID, CommunityArea), results(InspectionID, 1)), PassedInspections),length(PassedInspections, Count)")
    #print(list(prolog.query("passed_inspections_in_area(community_area('albany park'), Count)")))
    #Clausola per trovare la percentuale di ispezioni passate in un'area
    prolog.assertz("percentage_passed_inspections_in_area(CommunityArea, Percentage) :-total_inspections_in_area(CommunityArea, TotalCount),passed_inspections_in_area(CommunityArea, PassedCount),TotalCount > 0, Percentage is (PassedCount / TotalCount) * 100")
    #print(list(prolog.query("percentage_passed_inspections_in_area(community_area('albany park'), Percentage)")))
    #Clausola per trovare il numero di ispezioni passate con condizione in un'area
    prolog.assertz("passed_with_condition_inspections_in_area(CommunityArea, Count) :-findall(InspectionID, (inspection_in_community_area(InspectionID, CommunityArea), results(InspectionID, 2)), PassedWithConditionInspections),length(PassedWithConditionInspections, Count)")
    #print(list(prolog.query("passed_with_condition_inspections_in_area(community_area('albany park'), Count)")))
    #Clausola per trovare la percentuale di ispezioni passate con condizione in un'area
    prolog.assertz("percentage_passed_with_condition_inspections_in_area(CommunityArea, Percentage) :-total_inspections_in_area(CommunityArea, TotalCount),passed_with_condition_inspections_in_area(CommunityArea, PassedWithConditionCount),TotalCount > 0, Percentage is (PassedWithConditionCount / TotalCount) * 100")
    #print(list(prolog.query("percentage_passed_with_condition_inspections_in_area(community_area('albany park'), Percentage)")))
    #Clausola per calcolare un high risk index sula base del crime index e dell'health index
    prolog.assertz("high_risk_area(InspectionID) :-inspection_in_community_area(InspectionID, CommunityArea),crime_index(CommunityArea, CrimeIndex),health_index(CommunityArea, HealthIndex),CrimeIndex > 12,HealthIndex < 58")
    print(bool(list(prolog.query("high_risk_area(inspection_id(2582482))"))))
    

create_KB()