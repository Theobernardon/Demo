import pandas as pd
import csv

def csv_to_annotated_csv_pandas(input_csv, output_csv,
                                measurement_column,
                                tag_columns,
                                field_columns,
                                timestamp_column):
    """
    Convertit un fichier CSV en un CSV annoté compatible avec InfluxDB et Flux en utilisant Pandas.

    Le CSV annoté comporte trois lignes d'annotations :
      - #datatype : type de chaque colonne
      - #group : booléen indiquant si la colonne est utilisée pour le groupement (ici, "false" pour toutes)
      - #default : valeur par défaut (vide pour toutes)
    Puis la ligne d'en‐tête et enfin les données.

    Paramètres:
      - input_csv (str) : chemin du CSV source.
      - output_csv (str) : chemin du CSV annoté généré.
      - measurement_column (str) : colonne contenant le nom de la mesure.
      - tag_columns (list) : colonnes à utiliser comme tags.
      - field_columns (list) : colonnes à utiliser comme fields.
      - timestamp_column (str) : colonne contenant le timestamp au format ISO8601.
    """
    # Lecture du fichier CSV avec Pandas
    df = pd.read_csv(input_csv)
    
    # Récupération des noms de colonnes
    header = df.columns.tolist()
    
    # Déduire le type de chaque colonne pour l'annotation
    # Pour le timestamp : "dateTime:RFC3339"
    # Pour la mesure et les tags : "string"
    # Pour les fields : si la colonne est de type numérique, on choisit "double", sinon "string"
    datatypes = []
    for col in header:
        if col == timestamp_column:
            datatypes.append("dateTime:RFC3339")
        elif col == measurement_column or col in tag_columns:
            datatypes.append("string")
        elif col in field_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                datatypes.append("double")
            else:
                # On essaie de convertir la première valeur non nulle pour décider
                try:
                    first_val = df[col].dropna().iloc[0]
                    float(first_val)
                    datatypes.append("double")
                except Exception:
                    datatypes.append("string")
        else:
            datatypes.append("string")
    
    # La ligne de group : "false" pour toutes
    group_line = ["false"] * len(header)
    # La ligne default : vide pour toutes
    default_line = [""] * len(header)
    
    # Traitement du timestamp pour s'assurer qu'il soit au format ISO8601 (si possible)
    if timestamp_column in df.columns:
        try:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        except Exception:
            pass  # Si la conversion échoue, on laisse tel quel

    # Écriture du fichier annoté
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Écriture des trois lignes d'annotations
        writer.writerow(["#datatype"] + datatypes)
        writer.writerow(["#group"] + group_line)
        writer.writerow(["#default"] + default_line)
        # Écriture de la ligne d'en-tête
        writer.writerow(header)
        # Écriture des données sans en-tête (pour éviter de l'écrire deux fois)
        df.to_csv(f, index=False, header=False)