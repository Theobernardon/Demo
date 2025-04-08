import pandas as pd
from tqdm import tqdm
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Configuration de la connexion
token = "TOKEN_PRIVÉ"
org = "testinfluxdb2entreprise"
url = "http://localhost:8086"
csvfile = "data\KAG_energydata_complete.csv"
bucket = "importcsv"

# Initialisation du client
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

write_api = client.write_api(write_options=SYNCHRONOUS)

# Lecture du fichier CSV
df = pd.read_csv(csvfile, sep=",", decimal=".")
# Mise au bon format des données temporelles
df['date'] = pd.to_datetime(df['date'])


for id, raw in tqdm(df.iterrows(), desc="Writing data to InfluxDB", total=len(df)):
    # Formatage des données pour InfluxDB
    point_coso = (
        Point("Consommation")
        .field("Électroménager", raw['Appliances'])
        .field("Lumières", raw['lights'])
        .time(raw['date'], WritePrecision.S)
    )
    point_temp = (
        Point("Températures")
        .field("cuisine", raw['T1'])
        .field("salon", raw['T2'])
        .field("buanderie", raw['T3'])
        .field("extérieure", raw['T_out'])
        .time(raw['date'], WritePrecision.S)
    )
    
    point_hum = (
        Point("Humidité")
        .field("cuisine", raw['RH_1'])
        .field("salon", raw['RH_2'])
        .field("buanderie", raw['RH_3'])
        .field("extérieure", raw['RH_out'])
        .time(raw['date'], WritePrecision.S)
    )
    
    point_vent = (
        Point("Vitesse du vent")
        .field("Chièvres", raw['Windspeed'])
        .time(raw['date'], WritePrecision.S)
    )
    
    point_press = (
        Point("Pression")
        .field("Chièvres", raw['Press_mm_hg'])
        .time(raw['date'], WritePrecision.S)
    )
    
    point_visib = (
        Point("Visibilité")
        .field("Chièvres", raw['Visibility'])
        .time(raw['date'], WritePrecision.S)
    )
    
    points = [point_coso, point_temp, point_hum, point_vent, point_press, point_visib]
    for point in points:
        # Écriture des points dans InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)

client.close()
