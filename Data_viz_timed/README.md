# Démo InfluxDB avec ECharts & Grafana ⚡

Ce projet est une démonstration de l'utilisation d'**InfluxDB** avec **ECharts** et **Grafana** pour l'analyse énergétique sur des séries temporelles.

## Installation d’InfluxDB 2

### Mise en place du conteneur Docker

Dans le terminal Docker, exécutez la commande suivante :

```cmd
docker run -p 8086:8086 --name "influxdbdocker" -v "C:\Users\berna\Documents\Programation\Demo\Data_viz_timed\data:/var/lib/influxdb2" -v "C:\Users\berna\Documents\Programation\Demo\Data_viz_timed\config:/etc/influxdb2" influxdb:2
```

Cette commande crée un nouveau conteneur Docker avec InfluxDB 2. Dans cet exemple, les dossiers **`data`** et **`config`** du répertoire de travail sont utilisés pour garantir la persistance des données, même en cas de suppression ou de redémarrage du conteneur.  
Le port **8086** est également ouvert à la fois sur l’hôte et le conteneur pour permettre l’accès à l’interface web.

### Clé en lecture seule disponible

Si vous souhaitez télécharger le dépôt et tester le travail présenté dans cette démonstration :

> **Nom d’utilisateur** : `readuser`  
> **Mot de passe** : `readuserpwd`

### Initialisation de la base de données

Pour ce tutoriel, nous utiliserons l’interface utilisateur disponible à l’adresse [http://localhost:8086](http://localhost:8086).

Il est aussi possible de tout configurer via le CLI installé dans le conteneur Docker :

```sh
docker exec influxdb2 influx setup \
  --username $USERNAME \
  --password $PASSWORD \
  --org $ORGANIZATION \
  --bucket $BUCKET \
  --force
```

#### Gestion de la sécurité et des tokens

> Pour ce tutoriel, la sécurité et les droits d’accès ne seront pas abordés.

Cependant, pour un déploiement en production, il est fortement recommandé de gérer les droits d'accès et la sécurité via les tokens.  
Cela se configure dans l’onglet `Load Data` > `API Tokens` de l’interface.  
Le gestionnaire de tokens permet de personnaliser les permissions (lecture/écriture) par utilisateur ou cas d’usage.

### Importation des fichiers CSV

L’importation directe de fichiers CSV (au format annoté), bien que supportée dans certaines versions, n’est pas compatible avec la configuration utilisée pour ce tutoriel.

#### Outil CSV → CSV annoté

Ma première piste consistait à annoter automatiquement un fichier CSV classique.  
J’avais entamé la rédaction d’un script (`csvannotator.py`) permettant cette transformation, mais je ne l’ai ni finalisé ni testé en raison de problèmes d’environnement.  
Je le laisse néanmoins à disposition pour référence.

#### Méthode retenue pour l'importation des données

Finalement, j’ai opté pour l’utilisation de l’interface CLI du package Python `influxdb_client` :

```python
# Importation des modules nécessaires
import pandas as pd
from tqdm import tqdm
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
```

##### Connexion au client InfluxDB

```python
token = "TOKEN_PRIVÉ"
org = "testinfluxdb2entreprise"
url = "http://localhost:8086"
csvfile = "data\KAG_energydata_complete.csv"
bucket = "importcsv"

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)
```

##### Préparation des données avec Pandas

```python
df = pd.read_csv(csvfile, sep=",", decimal=".")
df['date'] = pd.to_datetime(df['date'])
```

##### Enregistrement ligne par ligne dans InfluxDB

Une boucle `for` est utilisée pour parcourir les lignes du DataFrame et écrire chaque point dans InfluxDB selon le modèle de base souhaité.  
Les mesures, champs et tags sont définis librement. Une date est associée à chaque point.

J’ai utilisé `tqdm` pour afficher une barre de progression, car l’opération était assez longue (environ 1h30 sur mon ordinateur).

```python
for id, raw in tqdm(df.iterrows(), desc="Writing data to InfluxDB", total=len(df)):
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
        write_api.write(bucket=bucket, org=org, record=point)

client.close()
```

### Test de Dashboard

Une fois les données importées, j’ai pu tester quelques fonctionnalités de la plateforme, notamment la création de dashboards :

![Exemple de Dashboard](/ressourcestuto/dashboardtest.png)

##  Installation de grafana
### Mise en place du docker

```cmd
docker run -d --name=grafana -p 3000:3000 grafana/grafana 
```


