# Démo InfluxDB avec ECharts & Grafana ⚡

Ce projet est une démonstration de l'utilisation d'InfluxDB avec ECharts et Grafana pour l'analyse énergétique sur des séries temporelles.

## Installation d'influxdb 2

### Mise en place du docker
Dans le terminal de docker on exécute la commande suivante:  

```cmd
docker run -p 8086:8086 --name "influxdbdoker" -v "C:\Users\berna\Documents\Programation\Demo\Data_viz_timed\data:/var/lib/influxdb2" -v "C:\Users\berna\Documents\Programation\Demo\Data_viz_timed\config:/etc/influxdb2" influxdb:2
```

Cette commande va créer un nouveau docker avec influxDB2, pour l'exemple j'ai pris les dossiers **\data** et **\config** de l'espace de travail en cours pour sauvegarder les données de façon pérenne (en cas de suppression ou de relance du container). Pour les besoins du fonctionnement d'influx DB, ous allons également ouvrir le port 8086 de l'hôte et du docker.

### Clé en lecture seul à disposition
Si vous voulez télécharger le repository et testé le travail effectué dans la démo:
>Username: **readuser**

>Password: **readuserpwd**

### Initialisation de la base de données

Pour ce tuto nous allons suivre les différentes étapes de l'interface utilisateur disponible à l'adresse http://localhost:8086.  
  
A savoir qu'il est également possible de le faire avec les commandes influxdb avec le CLI installé dans le conteneur :
```sh
docker exec influxdb2 influx setup \
  --username $USERNAME \
  --password $PASSWORD \
  --org $ORGANIZATION \
  --bucket $BUCKET \
  --force
```

#### Gestion de la sécurité et des tokens
>Pour ce tutoriel, cet élément ne sera pas pris en compte !  
Ceci dit pour un déploiement plus important il sera nécessaire de géré la sécurité et les droits d'accès des différents tokens / utilisateurs.  
Tout cela peut se gérer dans l'onglet `Load Data` > `API Tokens`.  
Le gestionnaire de token permettra d'individualiser la gestion des permissions de lecture et d'écriture pour chaque cas d'utilisation.

### Importation des CSV
L'importation directe des fichier CSV (format annoter), pris en compte dans certaines versions, n'est pas permise par la technologie ni les versions sélectionnées lors du tuto.

#### outil CSV -> CSV annoter
La première piste que j'ai envisagé pour importer les données était l'ajout d'un CSVet j'avais donc édité une première piste d'outils permettant d'annoter automatiquement les CSV.  
Je ne l'ai pas encore optimisé ni même testé puisque je n'arrivais pas à importer les CSV pour des raisons d'environnement. Je laisse tout de même le fichier `csvannotator.py` Contenant la première piste de script d'annotation.

#### Méthode d'enregistrement des Data
La méthode finalement sélectionné a été d'utiliser l'interface CLI du package Python `influxdb_client`:  
```python
# Importation des modules et packages
import pandas as pd
from tqdm import tqdm
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
```
  
##### Initialisation de la partie clients influxdb
```python
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
```

##### Initialisation de la partie Data avec Pandas
```python
# Lecture du fichier CSV
df = pd.read_csv(csvfile, sep=",", decimal=".")

# Mise au bon format des données temporelles
df['date'] = pd.to_datetime(df['date'])
```

##### Enregistrement ligne par ligne du CSV
Dans la dernière partie du code, on utilise une boucle for pour parcourir les lignes du datafram et enregistrer les points selon le modèle de base de données influxDB que l'on veut obtenir à la fin.  
Nous répartissons donc les mesures les champs et les tags comme il nous entend tout en rentrant à chaque itération la date pour chaque point.  
La procédure étant relativement longue j'ai utilisé tqm pour avoir une estimation de temps de l'enregistrement des données. (Cela a duré à peu près 01h30 sur mon ordinateur)
```python
for id, raw in tqdm(df.iterrows(), desc="Writing data to InfluxDB", total=len(df)):
    # Formatage des données pour InfluxDB
    point_coso = (
        Point("Consommation")
        .field("Électroménager", raw['Appliances'])
        .field("Lumières", raw['lights'])
        .time(raw['date'], WritePrecision.S)
    )
    ...
    point_visib = (
        Point("Visibilité")
        .field("Chièvres", raw['Visibility'])
        .time(raw['date'], WritePrecision.S)
    )
    
    points = [point_coso, ..., point_visib]
    for point in points:
        # Écriture des points dans InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)

# Fermeture de la session clients
client.close()
```

### Test de Dashboard
Une fois les données importées j'ai eu l'occasion de tester quelques fonctionnalités de la plateforme comme la conception de dashboard:

![Image de l'exemple de Dashboard](/ressourcestuto/dashboardtest.png)

##  Installation de grafana
### Mise en place du docker

```cmd
docker run -d --name=grafana -p 3000:3000 grafana/grafana 
```


