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
                             f"community_area({inspection_id}, '{row['Community Area Name']}')",
                             f"no_violations({inspection_id}, {row['No Violations']})",
                             f"violations_on_management_and_supervision({inspection_id}, {row['Violations on Management and Supervision']})",
                             f"violations_on_hygiene_and_food_security({inspection_id}, {row['Violations on Hygiene and Food Security']})",
                             f"violations_on_temperature_and_special_procedures({inspection_id}, {row['Violations on Temperature and Special Procedures']})",
                             f"violations_on_food_safety_and_quality({inspection_id}, {row['Violations on Food Safety and Quality']})",
                             f"violations_on_instrument_storage_and_maintenance({inspection_id}, {row['Violations on Instrument storage and Maintenance']})",
                             f"violations_on_facilities_and_regulations({inspection_id}, {row['Violations on Facilities and regulations']})"]
                             
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
    #Clausola per determinare se una struttura ha avuto come risultato pass e non ha violazioni
    prolog.assertz("passed_no_violations(InspectionID) :-results(InspectionID, 1),no_violations(InspectionID, 1)")
    #print(bool(list(prolog.query("passed_no_violations(inspection_id(2370179))"))))
    #Clausola per determinare se una struttura ha avuto almeno una violazione,considerata grave, o tra quelle di management e supervision, o igiene e sicurezza alimentare, o temperatura e procedure speciali
    prolog.assertz("serious_violations(InspectionID) :-violations_on_management_and_supervision(InspectionID, Count),Count > 0;violations_on_hygiene_and_food_security(InspectionID, Count),Count > 0;violations_on_temperature_and_special_procedures(InspectionID, Count),Count > 0")
    #print(bool(list(prolog.query("serious_violations(inspection_id(2370179))"))))
    #Clausola per determinare il numero di struttura con violazioni serie in un'area e che non hanno passato l'ispezione
    prolog.assertz("serious_violations_in_area(CommunityArea, Count) :-findall(InspectionID, (inspection_in_community_area(InspectionID, CommunityArea),serious_violations(InspectionID),results(InspectionID, 0)), SeriousViolations),length(SeriousViolations, Count)")
    #print(list(prolog.query("serious_violations_in_area(community_area('near west side'), Count)")))
    #Clausola per determinare la percentuale di strutture con violazioni serie in un'area e che non hanno passato l'ispezione
    prolog.assertz("percentage_serious_violations_in_area(CommunityArea, Percentage) :-total_inspections_in_area(CommunityArea, TotalCount),serious_violations_in_area(CommunityArea, SeriousViolationsCount),TotalCount > 0, Percentage is (SeriousViolationsCount / TotalCount) * 100")
    #print(list(prolog.query("percentage_serious_violations_in_area(community_area('near west side'), Percentage)")))
    #Clausola per determinare la media del crime index
    prolog.assertz("average_crime_index(Average) :-findall(CrimeIndex, crime_index(_, CrimeIndex), CrimeIndexes),sum_list(CrimeIndexes, Sum),length(CrimeIndexes, Count),Average is Sum / Count")
    #print(list(prolog.query("average_crime_index(Average)")))
    #Clausola per determinare se un'area ha un crime index superiore alla media
    prolog.assertz("high_crime_area(CommunityArea) :-crime_index(CommunityArea, CrimeIndex),average_crime_index(Average),CrimeIndex > Average")
    #print(bool(list(prolog.query("high_crime_area(community_area('burnside'))"))))
    #Clausola per determinare la media dell'health index
    prolog.assertz("average_health_index(Average) :-findall(HealthIndex, health_index(_, HealthIndex), HealthIndexes),sum_list(HealthIndexes, Sum),length(HealthIndexes, Count),Average is Sum / Count")
    #print(list(prolog.query("average_health_index(Average)")))
    #Clausola per determinare se un'area ha un health index inferiore alla media
    prolog.assertz("low_health_area(CommunityArea) :-health_index(CommunityArea, HealthIndex),average_health_index(Average),HealthIndex < Average")
    #print(bool(list(prolog.query("low_health_area(community_area('burnside'))"))))
    #Clausola per determinare se un'area è ad alto rischio sulla base dell'health index e del crime index
    prolog.assertz("high_risk_area(CommunityArea) :-high_crime_area(CommunityArea),low_health_area(CommunityArea)")
    #print(bool(list(prolog.query("high_risk_area(community_area('greater grand crossing'))"))))
    #Clausola per determinare la media del below poverty level
    prolog.assertz("average_below_poverty_level(Average) :-findall(BelowPovertyLevel, below_poverty_level(_, BelowPovertyLevel), BelowPovertyLevels),sum_list(BelowPovertyLevels, Sum),length(BelowPovertyLevels, Count),Average is Sum / Count")
    #print(list(prolog.query("average_below_poverty_level(Average)")))
    #Clausola per determinare se un'area ha un below poverty level superiore alla media
    prolog.assertz("high_below_poverty_level(CommunityArea) :-below_poverty_level(CommunityArea, BelowPovertyLevel),average_below_poverty_level(Average),BelowPovertyLevel > Average")
    #print(bool(list(prolog.query("high_below_poverty_level(community_area('near south side'))"))))
    #Clausola per determinare la media del per capita income
    prolog.assertz("average_per_capita_income(Average) :-findall(PerCapitaIncome, per_capita_income(_, PerCapitaIncome), PerCapitaIncomes),sum_list(PerCapitaIncomes, Sum),length(PerCapitaIncomes, Count),Average is Sum / Count")
    #print(list(prolog.query("average_per_capita_income(Average)")))
    #Clausola per determinare se un'area ha un per capita income inferiore alla media
    prolog.assertz("low_per_capita_income(CommunityArea) :-per_capita_income(CommunityArea, PerCapitaIncome),average_per_capita_income(Average),PerCapitaIncome < Average")
    #print(bool(list(prolog.query("low_per_capita_income(community_area('burnside'))"))))
    #Clausola per determinare la media della disoccupazione
    prolog.assertz("average_unemployment_rate(Average) :-findall(UnemploymentRate, unemployment(_, UnemploymentRate), UnemploymentRates),sum_list(UnemploymentRates, Sum),length(UnemploymentRates, Count),Average is Sum / Count")
    #print(list(prolog.query("average_unemployment_rate(Average)")))
    #Clausola per determinrre se un'area ha una disoccupazione superiore alla media
    prolog.assertz("high_unemployment_rate(CommunityArea) :-unemployment(CommunityArea, UnemploymentRate),average_unemployment_rate(Average),UnemploymentRate > Average")
    #print(bool(list(prolog.query("high_unemployment_rate(community_area('lake view'))"))))
    #Clausola per determinare se una'area è alto rischio economico considerando se ha un reddito inferiore alla media, una disoccupazione superiore, e soglia di povertà superiore
    prolog.assertz("high_economic_risk_area(CommunityArea) :-low_per_capita_income(CommunityArea),high_unemployment_rate(CommunityArea),high_below_poverty_level(CommunityArea)")
    #print(bool(list(prolog.query("high_economic_risk_area(community_area('near north side'))"))))
    

    
    




    

create_KB()