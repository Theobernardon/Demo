# ⚡ Démonstration InfluxDB + ECharts + Grafana pour l'Analyse Énergétique

Ce projet est une démonstration complète de la visualisation de données énergétiques à l’aide de **InfluxDB**, **Grafana**, et **ECharts**. Il repose sur un jeu de données réel concernant la consommation d'énergie domestique et les conditions météorologiques associées.

---

## 📦 Données utilisées

> **Source Kaggle :**  
> [Appliances Energy Prediction Dataset](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)

Le jeu de données couvre 4,5 mois de mesures prises toutes les 10 minutes dans une maison, ainsi que des données météorologiques issues de la station de l’aéroport de **Chièvres**, Belgique.

### 📊 Variables principales :

| Variable        | Description |
|-----------------|-------------|
| `Appliances`    | Consommation des appareils électroménagers (Wh) |
| `lights`        | Consommation des lumières (Wh) |
| `T1` à `T9`     | Températures dans différentes pièces (°C) |
| `RH_1` à `RH_9` | Humidité relative (%) |
| `To`            | Température extérieure (station météo) |
| `Pression`      | Pression atmosphérique (mm Hg) |
| `RH_out`        | Humidité extérieure (%) |
| `Windspeed`     | Vitesse du vent (m/s) |
| `Visibility`    | Visibilité (km) |
| `Tdewpoint`     | Point de rosée (°C) |
| `rv1`, `rv2`    | Variables aléatoires (adimensionnelles) |

---

## 📥 Installation d’InfluxDB 2

### 🐳 Déploiement d'un Docker
Dans un terminal Docker :

```bash
docker run -p 8086:8086 --name "influxdbdocker" \
-v "C:\path\to\data:/var/lib/influxdb2" \
-v "C:\path\to\config:/etc/influxdb2" influxdb:2
```

- Persistance des données via volumes `data` et `config`.
- Port `8086` exposé pour l’accès à l’interface web.

Accès à l’interface web via : [http://localhost:8086](http://localhost:8086)

### 🔐 Clé de lecture publique (démo)

Pour tester sans tout reconfigurer :

- **Utilisateur :** `readuser`  
- **Mot de passe :** `readuserpwd`

> Pour un usage production, il est recommandé de configurer les **tokens** / user dans `Load Data > API Tokens`.

---

### Initialisation de la base de données

Pour ce tutoriel, J'ai utilisé l’interface utilisateur disponible à l’adresse [http://localhost:8086](http://localhost:8086).

Il est aussi possible de tout configurer via le CLI installé dans le conteneur Docker :

```sh
docker exec influxdb2 influx setup \
  --username $USERNAME \
  --password $PASSWORD \
  --org $ORGANIZATION \
  --bucket $BUCKET \
  --force
```

## 🛠️ Importation des données avec Python

### 📦 Préparation

On utilise `pandas`, `tqdm`, et le client officiel InfluxDB :

```bash
pip install influxdb-client pandas tqdm
```

### 🐍 Exemple de script

Le script suivant a permis d'importer le CSV dans la base de données influxdb

```python
# Importation des modules nécessaires
import pandas as pd
from tqdm import tqdm
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
```

```python
# Connexion au client InfluxDB
token = "TOKEN_PRIVÉ"
org = "testinfluxdb2entreprise"
url = "http://localhost:8086"
csvfile = "data\KAG_energydata_complete.csv"
bucket = "importcsv"

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)
```

```python
# Préparation des données avec Pandas
df = pd.read_csv(csvfile, sep=",", decimal=".")
df['date'] = pd.to_datetime(df['date'])
```
Une boucle `for` est utilisée pour parcourir les lignes du DataFrame et écrire chaque point dans InfluxDB selon le modèle de base souhaité.  
Les mesures, champs et tags sont définis librement. Une date est associée à chaque point.

J’ai utilisé `tqdm` pour afficher une barre de progression, car l’opération était assez longue (environ 1h30 sur mon ordinateur).
```python
# Enregistrement ligne par ligne dans InfluxDB
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

> Le script `csvannotator.py` (incomplet) et le jupyter `test.ipynb` sont laissés pour référence si vous souhaitez générer des fichiers au format CSV annoté. (Je n'ai pas pu tester sur influxdb 2)

## 📊 Visualisation avec InfluxDB UI

Une fois les données importées, on peut créer des dashboards directement depuis l’interface web InfluxDB.

![Exemple de dashboard](/ressourcestuto/dashboardtest.png)

---

## 📈 Intégration de Grafana (à venir)

### ⚙️ Installation avec Docker

```bash
docker run -d --name=grafana -p 3000:3000 grafana/grafana
```

- Accès à Grafana via : [http://localhost:3000](http://localhost:3000)
- Ajouter **InfluxDB** comme source de données.
- Créer des visualisations dynamiques à partir des séries importées.

---

## 🌐 Intégration avec ECharts (à venir)

La prochaine étape consistera à intégrer **Apache ECharts** pour visualiser certaines séries temporelles dans une interface web personnalisée.

---
