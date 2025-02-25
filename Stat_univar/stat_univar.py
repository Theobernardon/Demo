import inspect
import base64
import re 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm
from datatheo.outil import format_numb, limite_char
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from matplotlib.offsetbox import AnchoredText
from datatheo.outil import ajust_coord
from io import BytesIO
from IPython.display import display, HTML
from abc import ABC, abstractmethod


#### Calcules ####
def gini_indice(serie, presision=20, centrer_sur_etendue=False, dropna=True, fillna=0, message=True):
    """Calcul de l'indice de Gini:
    - Plus il est proche de 0, plus la répartition est égalitaire
    - Plus il est proche de 1, plus la répartition est inégalitaire
    Aproximation de l'air sous la courbe utiliser:
    calcul de l'air sous la courbe par l'adition de l'aire d'un 
    triangle et de l'aire de X trapèses. X dépend de la présision.
    Plus la présision augmente, plus il y aura de trapèse dans le
    calcul, plus il sera juste. (20 par defaut)

    Args:
        serie (pd.series): colonne sur la quelle faire l'annalyse
        presision (int, optional): definit le nombre de séparateure sur la courbe pour le calcul de l'air. Defaults to 20.

    Returns:
        float: indice de Gini
    """
    # Gestion des Nans
    nb_nans = serie.isna().sum()
    if nb_nans != 0:
        if not dropna:
            serie.fillna(fillna, inplace=True)
            if message:
                print(f"ATTENTION !\nLes {nb_nans} Nans \nont été remplacés par '{fillna}'")
        else:
            serie.dropna(inplace=True)
            if message:
                print(f"ATTENTION !\nLes {nb_nans} Nans \nont été supprimés")
    
    #### Création de la table ####
    nom_de_serie = serie.name
    # Mise en variable de la serie
    table_lorenz = serie
    # Tris des données
    table_lorenz = table_lorenz.sort_values()
    # Si centrer_sur_etendue = True alors soustraction du minnimum a toutes les valeures:
    if centrer_sur_etendue:
        table_lorenz = table_lorenz.apply(lambda x: x - table_lorenz[0])
    # Transformation en somme cummulée:
    table_lorenz = table_lorenz.cumsum()
    # Transformation en fréquances cumulées (axe y)
    table_lorenz = table_lorenz / table_lorenz.iloc[-1]
    # Réinitialisation de l'index (au cas ou c'est un str)
    table_lorenz = table_lorenz.reset_index(drop=True)
    # Transformation de l'id en proportion de population (axe x)
    table_lorenz.index = (table_lorenz.index + 1) / (table_lorenz.index[-1] + 1)

    # Sélection des sections pour les calcules d'aires
    # Aproximation: découpage en aire d'un triangle et X trapèses
    table_lorenz = table_lorenz.reset_index()
    # Sélection des lignes
    nombre_de_lignes_a_sauter = int(table_lorenz.shape[0] / presision)
    nombre_de_lignes_a_sauter = nombre_de_lignes_a_sauter if nombre_de_lignes_a_sauter >= 1 else 1
    liste_des_lignes_selectioner = list(range(0, table_lorenz.shape[0], nombre_de_lignes_a_sauter)) +  [table_lorenz.index[-1]]
    # Calcul de l'air sous la courble
    air_sous_la_courbe = (table_lorenz.loc[0, nom_de_serie] * table_lorenz.loc[0, "index"]) / 2
    for id_inf, id_sup in zip(liste_des_lignes_selectioner, liste_des_lignes_selectioner[1:]):
        hauteur = table_lorenz.loc[id_sup, "index"] - table_lorenz.loc[id_inf, "index"]
        petite_base, grande_base = table_lorenz.loc[id_inf, nom_de_serie],  table_lorenz.loc[id_sup, nom_de_serie]
        air = ((petite_base + grande_base) * hauteur) / 2
        air_sous_la_courbe += air
    # Calcul de l'air de la surface_de_concentration, entre la répartion parfaitement égalitaire et l'expérimental
    surface_de_concentration = 0.5 - air_sous_la_courbe
    # Calcul de l'indice de gini
    gini = surface_de_concentration * 2
    return round(gini, 5)


#### Représentation graphiques ####
def Edite_fig_or_axe(function):
    # Obtenir la signature de la fonction
    signature = inspect.signature(function)
    # Obtenir les noms des paramètres positionnels ou nommés
    param_names = [param.name for param in signature.parameters.values() 
                   if param.kind == param.POSITIONAL_OR_KEYWORD]

    def fonction_modifier(*args, **kwargs):
        # Créer un dictionnaire pour stocker les arguments
        args_dict = dict(zip(param_names, args))
        
        # Ajouter les arguments nommés au dictionnaire
        args_dict.update(kwargs)
        
        # Fusionner les dictionnaires des arguments et des paramètres par défaut
        param = {**{param.name: param.default for param in signature.parameters.values()
                    if param.default is not param.empty}, **args_dict}

        # Gestion de l'intégration dans un axe
        if param['ax'] is None:
            fig, param['ax'] = plt.subplots(figsize=param.get('figsize'))

        # Appeler la fonction avec les arguments fusionnés
        function(**param)
    return fonction_modifier

@Edite_fig_or_axe
def courbe_lorenz(serie, hue=None, ax=None, figsize=(7, 7), title="", text_frontsize=12,
                  centrer_sur_etendue=False, xticks=[], yticks=[], dropna=True, fillna=0):
    # https://progr.interplanety.org/en/python-how-to-find-the-polygon-center-coordinates/ 2022-09-26 13:02
    # Fonction pour l'affichage du label de la surface entre les courbes
    def centroid(vertexes):
        _len = len(vertexes)
        _x_list = [vertex [0] for vertex in vertexes] + list(np.linspace(0, 1, _len))
        _y_list = [vertex [1] for vertex in vertexes] + list(np.linspace(0, 1, _len))
        _x = sum(_x_list) / (_len*2)
        _y = sum(_y_list) / (_len*2)
        return(_x, _y)
    
    def table_pour_lorenz(table_lorenz, centrer_sur_etendue=False):
        #### Création de la table ####
        # Tris des données
        table_lorenz = table_lorenz.sort_values()
        # Si centrer_sur_etendue = True alors soustraction du minnimum a toutes les valeures:
        if centrer_sur_etendue:
            table_lorenz = table_lorenz.apply(lambda x: x - table_lorenz[0])
        # Transformation en somme cummulée:
        table_lorenz = table_lorenz.cumsum()
        # Transformation en fréquances cumulées (axe y)
        table_lorenz = table_lorenz / table_lorenz.iloc[-1]
        # Réinitialisation de l'index (au cas ou c'est un str)
        table_lorenz = table_lorenz.reset_index(drop=True)
        # Transformation de l'id en proportion de population (axe x)
        table_lorenz.index = (table_lorenz.index + 1) / (table_lorenz.index[-1] + 1)
        return table_lorenz

    # Ecriture du titre auto si non présisée
    if title == "":
        title=f"Courbe de lorenz de: {serie.name}"
    
    # Gestion des Nans
    def gestion_na(serie, dropna, fillna, cumulna=0):
        nb_nans = serie.isna().sum()
        cumulna += nb_nans
        text_na_gestion = ""
        if nb_nans != 0:
            if not dropna:
                serie = serie.fillna(fillna)
                text_na_gestion = f"ATTENTION !\nLes {cumulna} Nans\nont été remplacés par '{fillna}'"
            else:
                serie.dropna(inplace=True)
                text_na_gestion = f"ATTENTION !\nLes {cumulna} Nans\nont été supprimés"
        return serie, text_na_gestion, cumulna
    
    #### Création de la courbe ####
    # Set les axes
    ax.margins(0.002)
    xticks += [0, 0.2, 0.4, 0.6, 0.8, 1]
    yticks += [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)
    
    # Tracer la droite central
    color=None
    if hue is not None:
        color=[0.1750865648952205, 0.11840023306916837, 0.24215989137836502]
    ax.plot([0, 1], [0, 1], color=color)
    
    if hue is None:
        #### Création de la table ####
        # Gestion des Nans
        serie, text_na_gestion, _ = gestion_na(serie, dropna=dropna, fillna=fillna)
        # Mise en variable de la serie
        table_lorenz = table_pour_lorenz(serie, centrer_sur_etendue=centrer_sur_etendue)
        # Tracer la courbe de lorenz
        table_lorenz.plot.line(x="index", figsize=figsize, ax=ax, grid=True, fontsize=text_frontsize*0.7) 
        # Dessin de l'air de S, l'aire entre la courbe de lorenz et la coourbe d'equirépartition
        # Récupération des coordonnées x,y de la courbe de lorenz pour le tracée du polygone (l'aire)
        ix = table_lorenz.index
        iy = table_lorenz
        # Complaition des coordonnées par l'origine et le point 1,1
        coordonnées = [(0, 0), *zip(ix, iy), (1, 1)]
        # Création du polygone
        poly = Polygon(coordonnées, facecolor='0.8', edgecolor='0.5')
        # Insqertion du ploygone sur la figure
        ax.add_patch(poly)
        # Coordonées du centroide
        centroid_poly = centroid(coordonnées)
        # Calcul et affichage de S, la surfface entre les courbes
        ax.text(centroid_poly[0], centroid_poly[1], f"S = {round(gini_indice(serie, message=False)/2, 3)}",
                verticalalignment='center', horizontalalignment='center', fontsize=text_frontsize*1.2)
    else:
        liste_hue_unique = hue.unique()
        ax_legende = ["Répartition égalitaire"]
        datafram = pd.concat([serie, hue], axis=1)
        serie_name = serie.name
        hue_name = hue.name
        cumulna = 0
        for categories, colore in zip(liste_hue_unique, sns.cubehelix_palette(len(liste_hue_unique), dark=0.3)):
            #### Création de la table ####
            # Gestion des Nans
            serie, text_na_gestion, cumulna = gestion_na(datafram.loc[datafram[hue_name] == categories, serie_name], dropna=dropna, fillna=fillna, cumulna=cumulna)
            # Mise en variable de la serie
            table_lorenz = table_pour_lorenz(serie, centrer_sur_etendue=centrer_sur_etendue)
            # Tracer la courbe de lorenz
            table_lorenz.plot.line(x="index", figsize=figsize, ax=ax, grid=True, fontsize=text_frontsize*0.7).get_lines()[-1].set_color(colore)
            # Mise a jour des legendes
            ax_legende.append(str(categories))
            # Dessin de l'air de S, l'aire entre la courbe de lorenz et la coourbe d'equirépartition
            # Récupération des coordonnées x,y de la courbe de lorenz pour le tracée du polygone (l'aire)
            ix = table_lorenz.index
            iy = table_lorenz
            # Complaition des coordonnées par l'origine et le point 1,1
            coordonnées = [(0, 0), *zip(ix, iy), (1, 1)]
            # Création du polygone
            s = f"S = {round(gini_indice(serie, message=False)/2, 3)}"
            poly = Polygon(coordonnées, facecolor=colore, edgecolor=colore, alpha=0.15)
            # Insqertion du ploygone sur la figure
            ax.add_patch(poly)
            # Mise a jour des legendes
            ax_legende.append(s)
        # Afficher la legende
        ax.legend(ax_legende, fontsize=text_frontsize*0.75)
    
    # set title
    ax.set_title(title, fontdict=dict(size=text_frontsize*1.2)) 
    
    # Text de la gestion des Nans
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white")
    ax.text(0.5, 0.95, text_na_gestion, bbox=bbox_props,
            verticalalignment='top', horizontalalignment='center', fontsize=text_frontsize, color='red')

@Edite_fig_or_axe
def pie(serie, hue=None, funct=[], funct_label=[], size = 'auto', titre='Piechart', text_frontsize=12, padding_label=0.08,
        figsize=(6,6), ax=None, kwargs_pie={}, limite_char_label=15, nb_part_max=10, limite_affiche_pct=3):
    # fontions préenregistrées
    if funct == 'uniques':
        def nb_multibles(serie):
            return serie.duplicated(keep=False).sum()
        def nb_unique(serie):
            return serie.shape[0] - serie.duplicated(keep=False).sum()
        funct = [nb_multibles, nb_unique]
        funct_label = [lambda x: f'Multiples\n({x})', lambda x: f'Uniques\n({x})']
    elif funct == 'nan':
        def count_nan(serie):
            return serie.isna().sum()
        def count_nonan(serie):
            return serie.shape[0] -serie.isna().sum()
        funct = [count_nan, count_nonan]
        funct_label = [lambda x: f'Nan\n({x})', lambda x: f'Values\n({x})']
    
    # Gestion de la taille de la police
    textprops = dict(fontsize=text_frontsize)
    
    # Gestion de la taille des camemberts
    if size == 'auto':
        if hue is None:
            size = 1
        else:
            size = 0.35
    
    # instancie l'ajustement des labels
    adjust = ajust_coord(padding_label)
    
    # Gestion de l'utilisation du pie avec ou sans fonction et avec ou sans hue
    Nb_func = len(funct)
    # Pas de hue mais fonctions
    if (hue is None) and funct:
        # On génère la série agréger en lui passant la liste de fonctions à réaliser sur les données
        df_pie = serie.aggregate(funct)
        
        # Pour chaque fonction, on rentre la valeur et le label. 
        # (Le label est une fonction qui prend en paramètre la valeur pour l'écrire en toutes lettres)
        labels_2 = []
        data_2 = []
        for i, val in zip(range(Nb_func), df_pie.values):
            labels_2.append(funct_label[i](val))
            data_2.append(val)
        # On règle la distance des pourcentages en fonction de la size
        pctdistance_2 = max([(1-size/2), 0.75])
        # On fait une liste du nombre de fois que l'on veut la séparation de 0.01 entre les parts de camembert
        explode = [0.01] * Nb_func
        
        # On crée le diagramme en camembert à partir de toutes les données et paramètres
        wedges, texts, ptc = ax.pie(data_2, labels=labels_2,
            wedgeprops=dict(width=size, edgecolor='w'), explode=explode,
            autopct='%1.1f%%', pctdistance=pctdistance_2, textprops=textprops, 
            **kwargs_pie)
        
    # Pas de hue et pas de fonction à appliquer
    elif (hue is None) and not funct:
        # Les données sont réparties par un simple value count
        dt_pie = serie.value_counts()
        # On ne sélectionne que les x premières valeurs
        top_counts = dt_pie.head(nb_part_max-1)
        # Et on enregistre toutes les autres dans une catégorie autre
        autres = dt_pie.sum() - top_counts.sum()
        # On assemble les 2 pour recréer un nouveau data frame
        top_pie = pd.concat([top_counts, pd.Series(autres, index=["Autres"])])
        
        # Réglage des distances et des espacements entre les parts de camembert
        pctdistance_2 = max([(1-size/2), 0.75])
        explode = [0.01] * len(top_pie)
        
        # On crée le diagramme en camembert à partir de toutes les données et paramètres
        wedges, texts, ptc = ax.pie(top_pie, labels=top_pie.index,
            wedgeprops=dict(width=size, edgecolor='w'), explode=explode,
            autopct='%1.1f%%', pctdistance=pctdistance_2, textprops=textprops, 
            **kwargs_pie)
        
    # hue et une fonction
    elif (hue is not None) and funct:
        # On ajoute à la liste des fonctions la fonction count
        funct.append('count')
        # On applique le tri de la série par le hue en appliquant les agrégations prédéfinies 
        # Créant ainsi un nouveau datafram Dont chaque ligne est un élément de hue, 
        # et chaque colonne représente une des fonctions
        df_pie = serie.groupby(hue, observed=False).aggregate(funct)
        # Pour chacune des agrégations que l'on a réalisées sur les différentes catégories, 
        # on Génère une nouvelle colonnequi comportera le label lié à la valeur d'agrégation
        for i, agg_cols in zip(range(Nb_func), df_pie.iloc[:, :-1].columns):
            df_pie.loc[:, f'labels_{i}'] = df_pie[f"{agg_cols}"].apply(funct_label[i])
        
        # Les colonnes de label seront automatiquement les x dernières avec x est égal le nombre de fonctions
        labels_1 = np.array(df_pie.iloc[:, np.arange(-Nb_func, 0, 1)]).flatten()
        # Les colonnes de données seront automatiquement les x premièreavec x étant le nombre de fonctions
        data_1 = np.array(df_pie.iloc[:, np.arange(0, Nb_func, 1)]).flatten()
        # Gestion des couleurs et des distances de pourcentage
        colors_1 = sns.cubehelix_palette(len(data_1), dark=0.3)
        pctdistance_1 = max([(1-size/2), 0.75])
        
        # Le 2e camembert sera celui du milieu, il représentera les value count qui ont été créés plus haut
        labels_2 = df_pie.index
        data_2 = df_pie['count']
        # Gestion des couleurs et des distances de pourcentage et label
        colors_2 = sns.cubehelix_palette(len(data_2), dark=0.3)
        labeldistance_2 = max([(1-size*2), 0.25])
        pctdistance_2 = max([(1-size*0.7), 0.65])
        
        # On crée le diagramme 2 en camembert à partir de toutes les données et paramètres
        wedges, texts, ptc =  ax.pie(data_2, labels=labels_2, labeldistance=labeldistance_2, radius=1-size, colors=colors_2,
            wedgeprops=dict(width=size, edgecolor='w'), textprops=textprops, 
            autopct='%1.1f%%', pctdistance=pctdistance_2, **kwargs_pie)
        
        # réglage de l'affichage 
        for i, pourcent in enumerate(ptc):
            if float(pourcent.get_text()[:-1]) < limite_affiche_pct:
                ptc[i].set_text("")
                texts[i].set_text("")
        for i, text in enumerate(texts):
            if len(text.get_text()[:-1]) > limite_char_label:
                new_text = text.get_text()[:-1][:limite_char_label] + " ..."
                texts[i].set_text(new_text)
                
        # On crée le diagramme 1 en camembert à partir de toutes les données et paramètres
        wedges, texts, ptc =  ax.pie(
                            data_1, labels=labels_1, labeldistance=1.15, radius=1, colors=colors_1,
                            wedgeprops=dict(width=size, edgecolor='w'), 
                            autopct='%1.1f%%', textprops=textprops, 
                            pctdistance=pctdistance_1, **kwargs_pie
                            )
        
    # hue et pas de fonctions
    elif (hue is not None) and not funct:
        # S'il n'y a pas de fonction ça veut dire que l'on veut un camembert
        # de répartition des comptes en fonction des 2 catégories
        df_pie = serie.groupby([hue, serie], observed=False).aggregate(['count']).sort_values('count', ascending=False)
        
        labels_2 = hue.unique()
        data_2 = [df_pie.loc[val].sum().iloc[0] for val in labels_2]
        colors_2 = sns.cubehelix_palette(len(data_2), dark=0.3)
        labeldistance_2 = max([(1-size*2), 0.25])
        pctdistance_2 = max([(1-size*0.7), 0.65])
        
        labels_1 = []
        data_1 = []
        for val in labels_2:
            top_counts = df_pie.loc[val].head(5)
            top_counts.loc["Autres"] = df_pie.loc[val].sum().iloc[0] - top_counts.sum().iloc[0]
            [labels_1.append(item) for item in top_counts.index]
            [data_1.append(item[0]) for item in top_counts.values]
        colors_1 = sns.cubehelix_palette(len(data_1), dark=0.3)
        pctdistance_1 = max([(1-size/2), 0.75])
        
        wedges, texts, ptc =  ax.pie(
                                data_2, labels=labels_2, labeldistance=labeldistance_2, radius=1-size,
                                colors=colors_2, wedgeprops=dict(width=size, edgecolor='w'), textprops=textprops, 
                                autopct='%1.1f%%', pctdistance=pctdistance_2, **kwargs_pie
                                )
        
        # réglage de l'affichage 
        for i, pourcent in enumerate(ptc):
            if float(pourcent.get_text()[:-1]) < limite_affiche_pct:
                ptc[i].set_text("")
                texts[i].set_text("")
        for i, text in enumerate(texts):
            if len(text.get_text()[:-1]) > limite_char_label:
                new_text = text.get_text()[:-1][:limite_char_label] + " ..."
                texts[i].set_text(new_text)

        wedges, texts, ptc =  ax.pie(
                            data_1, labels=labels_1, labeldistance=1.15, radius=1, colors=colors_1,
                            wedgeprops=dict(width=size, edgecolor='w'),
                            autopct='%1.1f%%', textprops=textprops, 
                            pctdistance=pctdistance_1, **kwargs_pie
                            )
    
    # réglage de l'affichage 
    for i, pourcent in enumerate(ptc):
        if float(pourcent.get_text()[:-1]) < limite_affiche_pct:
            ptc[i].set_text("")
            texts[i].set_text("")
    
    for i, text in enumerate(texts):
        adjust.rec(*texts[i].get_position())
        x_y, aligne = adjust.adjust()
        texts[i].set_position(x_y)
        texts[i].set_horizontalalignment(aligne[0])
        if len(text.get_text()[:-1]) > limite_char_label:
            new_text = text.get_text()[:-1][:limite_char_label] + " ..."
            texts[i].set_text(new_text)
    
    ax.set_title(titre, fontdict=dict(size=text_frontsize*1.2))

@staticmethod
def Df_decopose_to_serie(function):
    """
    Un décorateur qui transforme un DataFrame ou une série Pandas 
    en plusieurs séries, et applique une fonction donnée à chaque série 
    tout en tenant compte d'une variable de teinte (hue).
    
    Parameters:
    function (callable): La fonction à appliquer aux séries découpées.
    
    Returns:
    function: Une nouvelle fonction qui applique la fonction spécifiée 
                à chaque série du DataFrame.
    """
    def fonction_modifier(self, *args, **kwargs):
        """
        Fonction modifiée qui gère la décomposition des données du DataFrame 
        et l'application de la fonction fournie.
        
        Parameters:
        self: Référence à l'instance de la classe (utilisée si la méthode 
                est appelée à partir d'une instance).
        *args: Arguments positionnels, où le premier argument doit être un 
                DataFrame ou une série Pandas, et le second est la teinte (hue).
        **kwargs: Arguments nommés à passer à la fonction décorée.
        """
        datafram = args[0]  # Le premier argument doit être un DataFrame ou une série

        if isinstance(datafram, pd.DataFrame):
            hue = args[1]  # La deuxième argument, qui représente la teinte
            args = args[2:]  # Les autres arguments positionnels à transmettre

            hue_name = hue.name if hue is not None else None  # Nom de la teinte, s'il est fourni
            
            # Parcours des colonnes du DataFrame
            for serie_name in tqdm(datafram.columns, unit='series'):
                # Vérification du type de la colonne (booléen ou autre)
                if is_bool_dtype(datafram[serie_name]):
                    serie = datafram[serie_name].astype(str)  # Convertit en chaîne si booléen
                else:
                    serie = datafram[serie_name]  # Sinon, utilise la série telle quelle
                
                # Si la colonne correspond au nom de la teinte, on passe
                if serie_name == hue_name:
                    pass
                else:
                    # On vérifie que la colonne n'est pas vide
                    if datafram[serie_name].isna().all():
                        print(f"La colonne {serie_name} est vide")
                    else:
                        try:
                            # Applique la fonction sur la série, la teinte et les autres arguments
                            function(self, serie, hue, *args, **kwargs)
                        except Exception as e:
                            print(f"La série {serie_name} n'a pas pus être traitée:\nERROR: {e}\n")
        elif isinstance(datafram, pd.Series):
            # Si c'est une série, applique directement la fonction
            # On vérifie que la colonne n'est pas vide
                    if datafram.isna().all():
                        print(f"La colonne {datafram.name} est vide")
                    else:
                        try:
                            # Applique la fonction sur la série, la teinte et les autres arguments
                            function(self, *args, **kwargs)
                        except Exception as e:
                            print(f"La série {datafram.name} n'a pas pus être traitée:\nERROR: {e}\n")
        else:
            print('Le paramètre datafram doit être un DataFrame ou une série Pandas')
    return fonction_modifier  # Retourne la fonction modifiée

class AbstractHtmlReprPourFigure(ABC):
    
    def __init__(self, *args, **kwargs):
        self.text_html = None
        self.images_temp = dict()
        
        self._plot_save_img64(*args, **kwargs)
        self._edit_html()
    
    ### Quelques fonctions utilitaires ###
    def _image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _incementer_id_image(self, match):
        # Incrementation de l'identifiant de la balise image
        incrementid = int(match.group(0)[-1]) + 1
        self.id_image = match.group(0)[:-1] + str(incrementid)
        return  self.id_image
    
    ### Les méthodes à implanter ###
    @abstractmethod
    def _edit_html(self):
        '''Cette méthode doit éditer self.text_html:
        text_html Doit constituer le dashboard global.'''
        pass
    
    @abstractmethod
    def _edit_figure(self):
        '''Cette méthode doit éditer la figure avec MatplotLib (Ou compatible comme sns)
        Le premier paramètre doit être une série'''
        pass
    
    ### La visualisation HTML disponible dans Jupiter Notebook ###
    def _repr_html_(self):
        if bool(self.text_html):
            # Pour pouvoir afficher plusieurs fois dans un même Jupiter notebook la 
            # représentation nous devons modifier l'identifiant de la balise image 
            # ainsi que le nom de la fonction appelée par les boutons pour chaque REPR
            self.text_html = re.sub(f"{self.id_image}", self._incementer_id_image, self.text_html)
            # Puis on affiche la nouvelle représentation en texte HTML
            return display(HTML(self.text_html))
    
    ### Construction des images à partir de l'éditeur de figures ###
    @Df_decopose_to_serie
    def _plot_save_img64(self, serie, *args, **kwargs):
        self._edit_figure(serie, *args, **kwargs)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight') # Sauvegarde l'image dans le buffer
        plt.close() # Ferme la figure pour éviter d'en avoir la trace REPR
        buffer.seek(0)  # Repositionne le curseur au début du buffer
        image_png = buffer.getvalue()  # Récupère les données binaires de l'image
        buffer.close()  # Ferme le buffer

        # Encodage en Base64
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        self.images_temp[serie.name] = image_base64
    
    ### Fonctionnalités de sauvegarde de l'image HTML ###
    def save_html(self, file_name="auto"):
        first_col = self.datafram.columns[0]
        last_col = self.datafram.columns[-1]
        if file_name == "auto":
            if self.hue is None:
                file_name = f'Figure_analyse_univariées_de_{first_col}_a_{last_col}.html'
            else:
                file_name = f'Figure_analyse_univariées_de_{first_col}_a_{last_col}_par_{self.hue.name}.html'
        else:
            if not file_name.endswith(".html"):
                file_name = file_name + ".html"
        # Ouvrir le fichier en mode écriture
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(self.text_html)  # Écrire le contenu HTML dans le fichier
    
class stat_univariee(AbstractHtmlReprPourFigure):
    
    def __init__(self, datafram, hue=None, log_scale=None, hist_bins=30, figsize="auto", 
                 lorenz=True, lorenz_xticks=[], lorenz_yticks=[], text_frontsize=12, 
                 stat_centrales_lines=True, nombre_modes_max=2, ratio_boxplot=1, title="",  
                 rotation_label=35, secure=True, dropna_lorentz=True, fillna_lorentz=0):
        """
        Initialisation de l'objet.

        Parameters
        ----------
        datafram : pd.DataFrame or pd.Series
            Le datafram ou une série pandas. Si df, toutes les colonnes seront traitées.
        hue : pd.Series, optional
            Série de catégorie. Defaults to None.
        log_scale : bool or int, optional
            échelle logarithmique pour le kdeplot et l'histplot. Defaults to None.
        hist_bins : int, optional
            nombre de séparateurs sur l'histogramme. Defaults to 30.
        figsize : str, optional
            taille de la figure. Defaults to "auto".
        lorenz : bool, optional
            présence ou non de la partie courbe de lorenz. Defaults to True.
        lorenz_xticks : list, optional
            ajout de sections de grille sur la courbe de lorenz sur l'ax X. Defaults to [].
        lorenz_yticks : list, optional
            ajout de sections de grille sur la courbe de lorenz sur l'ax Y. Defaults to [].
        text_frontsize : int, optional
            taille de la police si hue actif. Defaults to 12.
        stat_centrales_lines : bool, optional
            affichage ou non des lignes de stats de tendances central sur le kdeplot. Defaults to True.
        nombre_modes_max : int, optional
            détermine le nombre au-dessus duquel les modes ne s'affichent plus. Defaults to 2.
        ratio_boxplot : int, optional
            change la largeur prise par le boxplot sur la figure (si hue). Defaults to 1.
        title : str, optional
            modifie le titre de la figure auto:" Annalyse univariée ... (n=nombre de données) ". Defaults to "Annalyses univariées de la série "{serie_name}" par catégorie "{hue}" (n=...)".
        rotation_label : int, optional
            rotation des labels sur l'axe X. Defaults to 35.
        secure : bool, optional
            vérifie si le hue n'a pas plus de 5 éléments. Defaults to True.
        dropna_lorentz : bool, optional
            gestion des Nan sur la courbe de lorenz. Defaults to True.
        fillna_lorentz : int or float, optional
            valeur de remplacement des Nan sur la courbe de lorenz si dropna_lorentz=False. Defaults to 0.

        Returns
        -------
        None
        """
        # verification du hue
        if (hue is not None) and secure:
            if (hue.nunique() >= 5):
                print("Pour des raisons de lisibilité, ne pas dépasser la limite de 5 valeurs différentes dans le hue")
                print("Pour désactiver cette sécurité, secure = False")
                pass
        
        self.datafram = datafram
        self.hue = hue
        self.legend_textsize_factor=0.7
        self.title_textsize_factor=1.2
        self.ticks_textsize_factor=0.7
        
        AbstractHtmlReprPourFigure.__init__(self, datafram, hue, log_scale, hist_bins, figsize, 
                                            lorenz, lorenz_xticks, lorenz_yticks, text_frontsize, 
                                            stat_centrales_lines, nombre_modes_max, ratio_boxplot, 
                                            title,  rotation_label, dropna_lorentz, fillna_lorentz)
    
    def _edit_html(self):
        if bool(self.images_temp):
            self.id_image = f"image{id(self)}0"
            # Déclaration du conteneur global
            self.text_html = f"""<!DOCTYPE html>
            <html lang="fr">
            <head>
                <meta charset="UTF-8">
                <title>Analyse univariée du dataframe</title>
            </head>
            <body>
            <div style="display:inline">"""
            # Déclaration des styles
            self.text_html += """
            <style> 
                div {background-color: rgb(31, 31, 31);} 
                button {margin: 0.5%;background-color: #696969;color: white;border: none;border-radius: 5px;padding: 5px 10px;cursor: pointer;} 
                img {position: relative; width: 100%; height: auto; max-height: 50vw; object-fit: contain;} 
            </style>"""
            
            # Déclaration du conteneur des boutons
            self.text_html += """<div style="display: inline-flexbox">"""
            
            # Construction des boutons en fonction des colonnes et des liens
            first_name = False
            for name, img64 in self.images_temp.items():
                if not first_name:
                    first_name = name
                self.text_html += f"""<button class="copyButton" onclick="changeImage{self.id_image}('data:image/jpeg;base64,{img64}')">{name}</button>"""
            
            # Image par défaut
            self.text_html += f"""
            </div>
            <div>
                <img id="{self.id_image}" src="data:image/jpeg;base64,{self.images_temp[first_name]}" alt="Image Display">
            </div>"""
            
            # JavaScript pour changer l'image
            self.text_html += f"""
            <script>
                function changeImage{self.id_image}"""
            self.text_html += """(imageUrl) {"""
            self.text_html += f"""document.getElementById('{self.id_image}').src = imageUrl;"""
            self.text_html += """}
            </script>
            <script>
            // Sélectionner tous les boutons avec la classe 'copyButton'
            const buttons = document.querySelectorAll(".copyButton");

            // Ajouter un événement de clic droit à chaque bouton
            buttons.forEach(button => {
                button.addEventListener("contextmenu", function(event) {
                    event.preventDefault();  // Empêche le menu contextuel par défaut

                    // Copier le contenu textuel du bouton dans le presse-papiers
                    navigator.clipboard.writeText(button.textContent)
                        .then(() => {
                            alert("Texte copié dans le presse-papiers !");
                        })
                        .catch(err => {
                            console.error("Échec de la copie du texte : ", err);
                        });
                });
            });
        </script>
            </div>
            </body>
            </html>"""

    def _mesures_stat_de_serie(self, serie, lorenz, dropna_lorentz, fillna_lorentz, nombre_modes_max):
        # Centrales (avec gestion de la posibilitée de série a plusieurs modes)
        moy = serie.mean()
        modes= serie.mode()
        median = serie.median()
        # Dispertion
        var =  serie.var()
        ecart_type =  serie.std()
        mini = serie.min()
        maxi = serie.max()
        etendue = maxi - mini
        # Applatissenment et symetrie
        skewness = serie.skew()
        kurtosis = serie.kurtosis()
        # Répartition
        gini = None
        if lorenz:                
            gini = gini_indice(serie, dropna=dropna_lorentz, fillna=fillna_lorentz, message=False)

        # Textes en fontion des mesures de formes (interprétation ou restriction)
        # Interprétation conditionelle de l'applatissement
        text_kurt = ""
        if kurtosis < 0:
            text_kurt = "distribution aplatie"
        elif kurtosis > 0:
            text_kurt = "distribution concentrée"
        else:
            text_kurt = "Distribution normale"
        # Interprétation conditionnelle de la symétrie
        text_skew = ""
        if skewness < 0:
            text_skew = "étalée à gauche"
        elif skewness > 0:
            text_skew = "étalée à droite"
        else:
            text_skew = "symétrique"
        # Messsage conditionelle de restriction à un nombre de modes
        # trop important et affichage du texte liée au mode
        if len(modes) > nombre_modes_max:
            # Si le nombre de modes dépasse la limite: message d'information
            ls_text_modes = [f"+ de {nombre_modes_max}",]
        elif len(modes) >= 1:
            # Si plusieures modes mais pas trop: controle du texte pour un afichage alignée
            ls_text_modes = [format_numb(value, 2) for value in modes]
        else:
            ls_text_modes = "ERR 0 modes"
        return moy, median, ls_text_modes, var, ecart_type, mini, maxi, etendue, modes, kurtosis, text_kurt, skewness, text_skew, gini
    
    def set_legend_size(self, ax, text_frontsize):
        # Vérifier si une légende a été créée
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("")
            for text in legend.get_texts():
                text.set_fontsize(text_frontsize * self.legend_textsize_factor)  # Ajuster la taille des étiquettes
        else:
            # Si la légende n'existe pas, la recréer avec les bonnes tailles
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # Vérifier que la liste n'est pas vide
                ax.legend(handles, labels, title="", prop={'size': text_frontsize * self.legend_textsize_factor})
        return ax

    def _edit_figure(self, serie, hue, log_scale, hist_bins, figsize, 
                     lorenz, lorenz_xticks, lorenz_yticks, text_frontsize, 
                     stat_centrales_lines, nombre_modes_max, ratio_boxplot, 
                     title, rotation_label, dropna_lorentz, fillna_lorentz):
        palette = None
        if (hue is not None) & (palette is None):
            palette = sns.cubehelix_palette(hue.nunique(), dark=0.4)
        
        mode_tronque = False
        
        # Gestion des Nans
        nb_nans = serie.isna().sum()
        text_na_gestion = ""
        if nb_nans != 0:
            if not dropna_lorentz:
                text_na_gestion = f"ATTENTION ! Les {nb_nans} Nans ont été remplacés par '{fillna_lorentz}'"
            else:
                text_na_gestion = f"ATTENTION ! Les {nb_nans} Nans ont été supprimés"
        
        #### Quentitatives ####
        if is_numeric_dtype(serie):
            
            min_val, max_val = serie.dropna().sort_values().iloc[[0, -1]].values
            if (min_val<0) and (max_val>0) and (lorenz == True):
                lorenz = False
                print("La série contient des valeurs négatives et positives\nLa courbe de lorenz est donc désactivée")
            
            mesures_stats_colore = []
            if hue is None:
                mesures_stats_colore.append(self._mesures_stat_de_serie(serie, lorenz, dropna_lorentz, fillna_lorentz, nombre_modes_max))
                mesures_stats_colore.append(None)
            else:
                datafram = pd.concat([serie, hue], axis=1)
                serie_name = serie.name
                hue_name = hue.name
                
                datafram.sort_values(by=hue_name, inplace=True)
                hue = datafram.loc[:, hue_name]
                serie = datafram.loc[:, serie_name]
                
                liste_hue_unique = hue.dropna().unique()
                for categories, colore in zip(liste_hue_unique, palette):
                    mesures_stats_colore.append(self._mesures_stat_de_serie(datafram.loc[datafram.loc[:, hue_name] == categories, serie_name], lorenz, dropna_lorentz, fillna_lorentz, nombre_modes_max))
                    mesures_stats_colore.append(colore)
            
            #### Création de la figure ####
            
            ### Figure ###
            # Instancie une figure avec la taille de l'option sans lorenz.
            # (si lorenz, la taille est ecraser par ça déclaration dans la fonction "courbe_lorenz")
            if figsize == 'auto':
                figsize = (16, 10)
            fig = plt.figure(figsize=figsize)

            ### Grille ###
            width_ratios = [(ratio_boxplot), 10, 6]
            # Instacie la separation principale de la figure en fontion de la présence ou non de la courbe de lorenz
            if lorenz:
                global_grid = gridspec.GridSpec(1, 3, figure = fig, width_ratios=width_ratios, wspace=0.15,)
            else:
                global_grid = gridspec.GridSpec(1, 2, figure = fig, width_ratios=width_ratios[0:2], wspace=0.15,)
            # Instacie la/les separation/s secondaires
            inner_grid_centre = global_grid[0, 1].subgridspec(2, 1, hspace=0.15,)
            if lorenz:
                inner_grid_gauche = global_grid[0, 2].subgridspec(2, 1, hspace=0.15, height_ratios=(2, 1))
            # Instacie les separation/s terciaires
            inner_grid_centre_hist = inner_grid_centre[1, 0].subgridspec(1, 2, wspace=0, hspace=0, width_ratios=(2, 1))
            inner_grid_centre_frec = inner_grid_centre[0, 0].subgridspec(1, 2, wspace=0, hspace=0, width_ratios=(2, 1))

            # La separation des grides est donc comme suit:
            # la figure est separer en 3 cases sur l'axe horizontal si lorenz ou deux si non
            # la première section reste entière et acceuillera le boxplot
            # la segonde est diviser en deux cases sur l'axe verticlal:
            #   - le haut acceuil la courbe des fréquances
            #   - le bas acceuil l'histograme 
            # chaqun d'eux est diviser en deux partie sur l'axe horizontal: 
            #   - à droite le graphique
            #   - à gauche un texte avec les valeures des mesures statistiques
            
            ### Axes ###
            # Atribution des diférents axes a leurs positions sur la grille comme dédrit ci-dessus
            # Boîte à moustaches
            ax_box = fig.add_subplot(global_grid[0, 0])
            # Histograme et son texte
            ax_hist_text = fig.add_subplot(inner_grid_centre_hist[0, 1])
            ax_hist_text.set_yticks([])
            ax_hist_text.set_xticks([])
            ax_hist = fig.add_subplot(inner_grid_centre_hist[0, 0])
            # Courbe de frequences et son texte
            ax_frec_text = fig.add_subplot(inner_grid_centre_frec[0, 1])
            ax_frec_text.set_yticks([])
            ax_frec_text.set_xticks([])
            ax_frec = fig.add_subplot(inner_grid_centre_frec[0, 0])
            # Courbe de lorenz et son texte (si lorenz)
            if lorenz:
                ax_lorenz_text = fig.add_subplot(inner_grid_gauche[1, 0])
                ax_lorenz_text.set_yticks([])
                ax_lorenz_text.set_xticks([])
                ax_lorenz = fig.add_subplot(inner_grid_gauche[0, 0])


            ### Tracer des graph ###
            ## Boîte à moustaches ##
            sns.boxplot(ax=ax_box, y=serie, hue=hue, width=0.5, palette=palette, legend=False)
            ax_box.set_title("Boîte à\nmoustaches", fontdict=dict(size=text_frontsize*self.title_textsize_factor))
            ax_box.tick_params(axis="both", labelsize=text_frontsize*self.ticks_textsize_factor)             
            
            ## Courbe de frequences et son texte ##
            # Courbe
            sns.kdeplot(ax=ax_frec, x=serie, hue=hue, fill=True, alpha=.5, log_scale=log_scale, palette=palette)
            ax_frec.set_title("Densité", fontdict=dict(size=text_frontsize*self.title_textsize_factor))
            ax_frec.set_ylabel("")
            ax_frec.set_xlabel("")
            ax_frec.tick_params(axis="both", labelsize=text_frontsize*self.ticks_textsize_factor)  
            # Text avec et sans catégories
            if hue is None:
                # Titre du paragraphe 1
                ax_frec_text.text(0.05, 0.9, "Tendance centrale:", fontsize=text_frontsize+3, ha='left', va='center')
                # Corps de texte 1
                ax_frec_text.text(0.05, 0.845, "\n".join([
                    f"Moyenne = {format_numb(mesures_stats_colore[0][0], 2)}",
                    f"Médiane = {format_numb(mesures_stats_colore[0][1], 2)}",
                    "\n".join(f"Mode/s = {mode}" if i == 0 else 
                              f"                {mode}" for i, mode in enumerate(mesures_stats_colore[0][2])),
                ]), fontsize=text_frontsize, ha='left', va='top')
                ax_frec.set_xlabel("")
                # Titre du paragraphe 2
                ax_frec_text.text(0.05, 0.5, "Dispersion:", fontsize=text_frontsize+3, ha='left', va='center')
                # Corps de texte 2
                ax_frec_text.text(0.05, 0.445, "\n".join([
                    f"Variance = {format_numb(mesures_stats_colore[0][3], 2)}",
                    f"Écart-type = {format_numb(mesures_stats_colore[0][4], 2)}",
                    f"Minimum = {format_numb(mesures_stats_colore[0][5], 2)}",
                    f"Maximum = {format_numb(mesures_stats_colore[0][6], 2)}",
                    f"Étendue = {format_numb(mesures_stats_colore[0][7], 2)}"
                ]), fontsize=text_frontsize, ha='left', va='top')
            else:
                nb_categ = len(liste_hue_unique)
                inner_grid_centre_frec_texts = inner_grid_centre_frec[0, 1].subgridspec(nb_categ, 1, wspace=0.025, hspace=0.025)
                for nu_categ in range(nb_categ):
                    inner_grid_centre_frec_texts_temp = inner_grid_centre_frec_texts[nu_categ, 0].subgridspec(1, 2, wspace=0.025, hspace=0.025)
                    ax_temp_central = fig.add_subplot(inner_grid_centre_frec_texts_temp[0, 0])
                    ax_temp_dispertion = fig.add_subplot(inner_grid_centre_frec_texts_temp[0, 1])
                    ax_temp = fig.add_subplot(inner_grid_centre_frec_texts[nu_categ, 0])
                    fancybox = mpatches.FancyBboxPatch(
                        [0.05, 0.05], 0.9, 0.9,
                        boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                        ec=mesures_stats_colore[(nu_categ*2)+1],
                        fc=mesures_stats_colore[(nu_categ*2)+1],
                        alpha=0.4
                        )
                    ax_temp.add_patch(fancybox)
                    ax_temp.set_axis_off()
                    text_central = "\n".join([f"Moy : {format_numb(mesures_stats_colore[nu_categ*2][0], 2)}",
                                            f"Med : {format_numb(mesures_stats_colore[nu_categ*2][1], 2)}",
                                            "\n".join(f"Mod : {mode}" if i == 0 else 
                                                      f"          {mode}" for i, mode in enumerate(mesures_stats_colore[nu_categ*2][2]))
                                            ])
                    ax_temp_central.text(0.065, 0.5, text_central, fontsize=text_frontsize, ha='left', va='center', in_layout=True)
                    ax_temp_central.set_axis_off()
                    text_dispertion = "\n".join([f"σ : {format_numb(mesures_stats_colore[nu_categ*2][4], 2)}",
                                                f"Min : {format_numb(mesures_stats_colore[nu_categ*2][5], 2)}",
                                                f"Max : {format_numb(mesures_stats_colore[nu_categ*2][6], 2)}"])
                    ax_temp_dispertion.text(0.065, 0.5, text_dispertion, fontsize=text_frontsize, ha='left', va='center', in_layout=True)
                    ax_temp_dispertion.set_axis_off()
            
            # Tracer des lignes de marcages des states de tendances centrales
            if stat_centrales_lines and (not hue is not None):
                # Moyen
                ax_frec.axvline(mesures_stats_colore[0][0], color="red", label="Moyenne")
                # Median
                ax_frec.axvline(mesures_stats_colore[0][1], color="orange", label="Médiane")
                # Modes (si le monbre de mode est inf ou egal au max)
                if  len(mesures_stats_colore[0][8]) <= nombre_modes_max & len(mesures_stats_colore[0][8]) > 1:
                    for mode in mesures_stats_colore[0][8]:
                        ax_frec.axvline(mode, color="green", label="Mode")
                elif len(mesures_stats_colore[0][8]) == 1:
                    ax_frec.axvline(mesures_stats_colore[0][8][0], color="green", label="Mode")
                # Affichage de la légende
                ax_frec = self.set_legend_size(ax_frec, text_frontsize)
            elif stat_centrales_lines:
                nb_categ = len(liste_hue_unique)
                for nu_categ in range(nb_categ):
                    # Moyen
                    ax_frec.axvline(mesures_stats_colore[nu_categ*2][0], color=mesures_stats_colore[(nu_categ*2)+1], label="Moyenne")
                    # Median
                    ax_frec.axvline(mesures_stats_colore[nu_categ*2][1], color=mesures_stats_colore[(nu_categ*2)+1], label="Médiane", ls='--')
                    # Modes (si le monbre de mode est inf ou egal au max)
                    if  len(mesures_stats_colore[nu_categ*2][8]) <= nombre_modes_max & len(mesures_stats_colore[nu_categ*2][8]) > 1:
                        for mode in mesures_stats_colore[nu_categ*2][8]:
                            ax_frec.axvline(mode,  color=mesures_stats_colore[(nu_categ*2)+1], label="Mode")
                    elif len(mesures_stats_colore[nu_categ*2][8]) == 1:
                        ax_frec.axvline(mesures_stats_colore[nu_categ*2][8][0], color=mesures_stats_colore[(nu_categ*2)+1], label="Mode", ls='-.')
                    # Affichage de la légende
                ax_frec.legend(prop={'size': text_frontsize * self.legend_textsize_factor})

            ## Histogramme et son texte ##
            sns.histplot(ax=ax_hist, x=serie, hue=hue, log_scale=log_scale, bins=hist_bins, palette=palette)
            ax_hist.set_title("Histogramme", fontdict=dict(size=text_frontsize*self.title_textsize_factor))
            ax_hist.set_ylabel("")
            ax_hist.tick_params(axis="both", labelsize=text_frontsize*self.ticks_textsize_factor) 
            ax_hist = self.set_legend_size(ax_hist, text_frontsize)
            if hue is None:
                # Titre du paragraphe 1
                ax_hist_text.text(0.05, 0.9, "Mesure d'aplatissement:", fontsize=text_frontsize+3, ha='left', va='center')
                # Corps de texte 1
                ax_hist_text.text(0.05, 0.78, "\n".join([
                    f"Kurtosis = {format_numb(mesures_stats_colore[0][9], 2)}",
                    f"{mesures_stats_colore[0][10]}",
                ]), fontsize=text_frontsize, ha='left', va='center')
                # Titre du paragraphe 2
                ax_hist_text.text(0.05, 0.6, "Mesure d'asymétrie:", fontsize=text_frontsize+3, ha='left', va='center')
                # Corps de texte 2
                ax_hist_text.text(0.05, 0.48, "\n".join([
                    f"Skewness = {format_numb(mesures_stats_colore[0][11], 2)}",
                    f"{mesures_stats_colore[0][12]}",
                ]), fontsize=text_frontsize, ha='left', va='center')
                # Corps de texte 3
                ax_hist_text.text(0.5, 0.22, f'"{mesures_stats_colore[0][10]}\net {mesures_stats_colore[0][12]}"', fontsize=text_frontsize+3, ha='center', va='center')
            else:
                nb_categ = len(liste_hue_unique)
                inner_grid_centre_hist_texts = inner_grid_centre_hist[0, 1].subgridspec(nb_categ, 1, wspace=0.025, hspace=0.025)
                for nu_categ in range(nb_categ):
                    ax_temp = fig.add_subplot(inner_grid_centre_hist_texts[nu_categ, 0])
                    fancybox = mpatches.FancyBboxPatch(
                        [0.05, 0.05], 0.9, 0.9,
                        boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                        ec=mesures_stats_colore[(nu_categ*2)+1],
                        fc=mesures_stats_colore[(nu_categ*2)+1],
                        alpha=0.4
                        )
                    ax_temp.add_patch(fancybox)
                    ax_temp.set_axis_off()
                    text_central = "\n".join([f"Kurt : {format_numb(mesures_stats_colore[nu_categ*2][9], 2)}" + " | " +
                                            f"Skew : {format_numb(mesures_stats_colore[nu_categ*2][11], 2)}",
                                            f'"{mesures_stats_colore[nu_categ*2][10]}\net {mesures_stats_colore[nu_categ*2][12]}"'
                                            ])
                    ax_temp.text(0.065, 0.5, text_central, fontsize=text_frontsize, ha='left', va='center', in_layout=True)



            ## Courbe de lorenz et son texte (si lorenz) ##
            if lorenz:
                if figsize == (16, 10):
                    figsize = (23, 10)
                courbe_lorenz(serie, hue=hue, ax=ax_lorenz, figsize=figsize, title="Courbe de Lorenz", text_frontsize=text_frontsize,
                                xticks=lorenz_xticks, yticks=lorenz_yticks, dropna=dropna_lorentz, fillna=fillna_lorentz)
                
                if hue is None:
                    # Titre du paragraphe
                    ax_lorenz_text.text(0.05, 0.85, "Mesure de répartition égalitaire:", fontsize=text_frontsize+3, ha='left', va='center')
                    # Corps de texte
                    ax_lorenz_text.text(0.05, 0.5, "\n".join([
                        f"Gini = {mesures_stats_colore[0][13]}\n",
                        "Plus il est proche de 0, plus la répartition est égalitaire",
                        "Plus il est proche de 1, plus la répartition est concertée",
                    ]), fontsize=text_frontsize, ha='left', va='center')
                    # Si nan gestion
                    ax_lorenz_text.text(0.05, 0.2, text_na_gestion, fontsize=text_frontsize, ha='left', va='center', color='red')
                
                else:
                    nb_categ = len(liste_hue_unique)
                    nb_rows = 2
                    nb_cols = (nb_categ + 1) // nb_rows
                    
                    inner_grid_gauche_text = inner_grid_gauche[1, 0].subgridspec(2, 1, wspace=0.025, hspace=0.025, height_ratios=(1,2))
                    inner_grid_gauche_text_haut = fig.add_subplot(inner_grid_gauche_text[0, 0])
                    inner_grid_gauche_text_haut.text(0.05, 0.5, "Mesure de répartition égalitaire:", fontsize=text_frontsize+3, ha='left', va='center')
                    inner_grid_gauche_text_haut.set_axis_off()
                    inner_grid_gauche_text_bas = inner_grid_gauche_text[1, 0].subgridspec(nb_rows, nb_cols, wspace=0.05, hspace=0.05,)
                    for nu_categ in range(nb_categ):
                        # Edition du texte
                        text_central = f"{liste_hue_unique[nu_categ]}:\n" + f"Gini = {mesures_stats_colore[nu_categ*2][13]}"
                        # Ajoutez la boîte de texte actuelle
                        grid_raw = 1 if nu_categ >= nb_cols else 0
                        grid_col = nu_categ % nb_cols  # Réorganiser les indices des colonnes
                        ax_temp = fig.add_subplot(inner_grid_gauche_text_bas[grid_raw, grid_col])
                        
                        fancybox = mpatches.FancyBboxPatch(
                            [0.05, 0.05], 0.9, 0.9,
                            boxstyle=mpatches.BoxStyle("Round", pad=0.05),
                            ec=mesures_stats_colore[(nu_categ*2)+1],
                            fc=mesures_stats_colore[(nu_categ*2)+1],
                            alpha=0.4
                        )
                        ax_temp.add_patch(fancybox)
                        ax_temp.set_axis_off()
                        
                        # Ajoutez le texte avec AnchoredText
                        at = AnchoredText(text_central, loc='center', frameon=False, prop=dict(fontsize=text_frontsize))
                        ax_temp.add_artist(at)
                        
            if hue is not None:
                ax_frec_text.set_axis_off()
                ax_hist_text.set_axis_off()
                if lorenz:
                    ax_lorenz_text.set_axis_off()
            
        else: 
            #### Qualitative ####
            
            #### création de la figure ####
            if figsize == 'auto':
                figsize = (21, 9)
            fig = plt.figure(figsize=figsize)
            
            text_frontsize = text_frontsize-2

            #### création de la grid ####
            global_grid = gridspec.GridSpec(1, 2, figure = fig, width_ratios=[1,3], wspace=0.15,)
            global_grid_droite = global_grid[0, 1].subgridspec(2, 1, hspace=0.3)
            grid_droite_haut = global_grid_droite[0, 0].subgridspec(1, 2)
            grid_droite_bas = global_grid_droite[1, 0].subgridspec(1, 2)

            #### création des axes ####
            ax_hcnt = fig.add_subplot(global_grid[0, 0])
            ax_hcnt.spines['right'].set_visible(False)

            ax_pie_nunique = fig.add_subplot(grid_droite_haut[0, 0])

            ax_pie_nans = fig.add_subplot(grid_droite_haut[0, 1])
            
            ax_droite_bas = fig.add_subplot(grid_droite_bas[0, :])
            ax_droite_bas.set_yticks([])
            ax_droite_bas.set_xticks([])
            ax_droite_bas.spines['top'].set_visible(False)
            ax_droite_bas.spines['bottom'].set_visible(False)
            ax_droite_bas.spines['left'].set_visible(False)
            ax_droite_bas.spines['right'].set_visible(False)

            ax_pie_top = fig.add_subplot(grid_droite_bas[0, 1])
            ax_pie_top.spines['top'].set_visible(False)
            ax_pie_top.spines['bottom'].set_visible(False)
            ax_pie_top.spines['left'].set_visible(False)
            ax_pie_top.spines['right'].set_visible(False)

            ax_hbar_top = fig.add_subplot(grid_droite_bas[0, 0])
            ax_hbar_top.spines['top'].set_visible(False)
            ax_hbar_top.spines['bottom'].set_visible(False)
            ax_hbar_top.spines['left'].set_visible(False)
            ax_hbar_top.spines['right'].set_visible(False)

            #### création des figures ####

            ### Countplot gauche ###
            # count plot 
            counts = serie.value_counts().head(500)
            ## Limiter au 500 premières occurrences (s'il y en a plus de 500)
            sns.barplot(ax=ax_hcnt, x=counts.values, y=counts.index.astype(str))
            ax_hcnt.set_title(f"Comptage valeurs de '{serie.name}' (limtée à 500)", fontdict=dict(size=text_frontsize*1.2))
            ax_hcnt.set_xlabel('Compte', fontdict=dict(size=text_frontsize))
            plt.setp(ax_hcnt.get_xticklabels(), rotation=rotation_label)
            if len(ax_hcnt.get_yticks()) >= 60:
                ax_hcnt.set_yticks([])
            elif len(ax_hcnt.get_yticks()) <= 10:
                fig.delaxes(ax_hcnt)
                mode_tronque = True

            ### Pieplot des valeures uniques ###
            size = 1
            if hue is not None:
                size = 0.35
                            
            pie(serie, hue, funct='uniques', titre='Répartition des valeurs uniques', text_frontsize=text_frontsize,
                size=size, kwargs_pie=dict( startangle=140), ax=ax_pie_nunique, limite_affiche_pct=0)
            
            ### Pieplot des Nans ###
            pie(serie, hue, funct='nan', titre='Répartition des Nans', limite_affiche_pct=0,
                size=size, kwargs_pie=dict( startangle=140), ax=ax_pie_nans, text_frontsize=text_frontsize)
            
            ### top 10 ###
            # Titre
            ax_droite_bas.text(0.5, 1.18, "Zoom sur Top 10", ha='center', va='top', fontsize=text_frontsize+3)

            ### Countplot top 10 ###
            # count plot top 10
            concat_serie_hue = pd.concat([serie, hue], axis=1)
            # Iddentification du top 10 des valeures les plus présentes de la série en cours
            index_top10 = serie.value_counts().head(10).sort_values(ascending=False).index
            # Dataframe, fusiondes 2 colonnes étudiées, en ne sélectionnant que les valeurs de la série principale les plus fréquentes
            concat_serie_hue = concat_serie_hue.loc[concat_serie_hue[serie.name].isin(index_top10)]
            # -->top 10 des valeurs de 'serie' et leurs hue
            # On isole le nom de la colonne hue dans une variable qui prendra la valeur None si hue a la valeur None
            hue_name = hue.name if hue is not None else None

            sns.countplot(concat_serie_hue, y=serie.name, hue=hue_name, ax=ax_hbar_top, palette=palette, order=index_top10)

            # Ajuster les étiquettes sur l'axe y pour qu'elles apparaissent à l'intérieur des barres
            liste_des_labels = [x.get_text() for x in ax_hbar_top.yaxis.get_ticklabels()]
            if hue is None:
                count_serie = serie.value_counts()
                for i, text in enumerate(liste_des_labels):
                    ax_hbar_top.text(count_serie[text] + count_serie.max() * 0.01, i, limite_char(text, 25), va='center', fontdict=dict(size=text_frontsize))
            else:
                # Accéder aux valeurs numériques pour chaque barre du countplot
                patches = ax_hbar_top.patches
                # On va construire une liste de tapple contenant les informations suivantes:
                # x, y, color, label (Le label n'étant pas présent partout naturellement il va falloir le récupérer grâce à la correspondance de couleur)
                coords = [(patch.get_width(), patch.get_y() + patch.get_height()/2, str(patch.get_facecolor()), patch.get_label()) for patch in patches]
                # Pour l'instant la variable coords Contient la liste des coordonnées qui nous 
                # intéressent et avec des coordonée de 0,0 la ref des Correspondance couleur label
                color_label_ref = []
                # On parcourt une copie de la liste
                for tup in coords.copy():
                    # Si les coordonnées sont 0,0 on l'ajoute à la liste color_ref et on la retire de la liste coordonnée
                    if tup[0] == 0 and tup[1] == 0:
                        color_label_ref.append(tup[2:])
                        coords.remove(tup)
                # Il ne reste plus qu'à transformer notre liste ref en dictionnaire
                color_label_ref = dict(color_label_ref)
                # Puis d'utiliser le dictionnaire pour transformer les couleurs en label
                coords = [(x,y,color_label_ref[color]) for x,y,color,label in coords]
                # On obtient donc tous les labels aux bonnes coordonnées
                for coord in coords:
                    ax_hbar_top.text(coord[0]*0.98, coord[1], limite_char(coord[2], 25), va='center', fontdict=dict(size=text_frontsize), horizontalalignment='right')
                count_serie = concat_serie_hue.value_counts()
                for i, text in enumerate(liste_des_labels):
                    ax_hbar_top.text(count_serie[text].max() + count_serie.max() * 0.01, i, limite_char(text, 25), va='center', fontdict=dict(size=text_frontsize))

            ax_hbar_top.set_title("Comptage", fontdict=dict(size=text_frontsize*1.2))
            ax_hbar_top.set_xlabel('Compte', fontdict=dict(size=text_frontsize))
            ax_hbar_top.set_yticklabels([])
            plt.setp(ax_hbar_top.get_xticklabels(), rotation=rotation_label)
            ax_hbar_top.grid(False)
            ax_hbar_top.legend([])

            # pie plot top 10
            pie(serie, hue, titre='Proportions de présence', text_frontsize=text_frontsize,
                size=size, kwargs_pie=dict(startangle=90), ax=ax_pie_top, nb_part_max=11)
            
        ## Titre de la figure ##
        if title == "":
            if hue is not None:
                title = f'Analyses univariées de la série "{serie.name}" par catégorie "{hue.name}" (n= {format_numb(serie.count())})'
            else:
                title = f'Analyses univariées de la série "{serie.name}" (n= {format_numb(serie.count())})'
        else:
            title = title + f" (n= {format_numb(serie.count())})"
        title = fig.suptitle(title, fontproperties= {'weight': 'semibold', 'size': text_frontsize+4})
        
        if mode_tronque:
            title.set(position=(title.get_position()[0]+1/8, 0.98))
        
        return