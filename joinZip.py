import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Carica il file GeoJSON delle community areas
community_areas = gpd.read_file("chicago-community-areas.geojson")
df = pd.read_csv("dataset/Food_Inspections_20240520.csv")

# Funzione per trovare la community area data una latitudine e longitudine
def find_community_area(lat, lon):
    point = Point(lon, lat)
    for idx, area in community_areas.iterrows():
        if area['geometry'].contains(point):
            return area['community']
    return None

# Esempio di coordinate (latitudine e longitudine)
coordinates = list(df[['Latitude', 'Longitude']].values)

# Associa ogni coppia di coordinate alla sua community area
for lat, lon in coordinates:
    community = find_community_area(lat, lon)
    print(f"Le coordinate ({lat}, {lon}) si trovano nella community area {community}")
