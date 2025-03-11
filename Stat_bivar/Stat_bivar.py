import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
import re
import inspect
import warnings
import pandas as pd
import matplotlib.patches as mpatches
import pandas.api.types as ptypes
from datatheo.outil import format_numb
from math import floor, ceil
from typing import Literal, Union, Optional
from IPython.display import display_html
from IPython.display import HTML
from scipy import stats as st
from rich.console import Console
from rich.theme import Theme

consl = Console(
        theme=Theme({
        "test": "#FBCE9E",
        "var": "#00BDC8 bold",
        "pval": "#00BDC8 bold",
        "h": "#7ACFB0",
        "chemin": "#F88F52 bold",
        "motclef": "#7ACFB0 underline",
        "choix": "#7ACFB0 underline bold",
        "espace": "#1C5588 bold"
        }, inherit=False), width=70)
    
#### Partie de visualisation ####
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

#### Partie de statistiques uni-variées ####
# servant aux choix des tests
def _interpretation_test(p, alpha, H0, H1):
    """Interprète le résultat d'un test d'hypothèse statistique.

    Paramètres:
    p (float): La p-value obtenue à partir du test.
    alpha (float): Le niveau de signification pour le test.
    H0 (str): L'énoncé de l'hypothèse nulle.
    H1 (str): L'énoncé de l'hypothèse alternative.

    Retourne:
    tuple: Un tuple contenant un booléen et une chaîne de caractères.
        Le booléen est False si l'hypothèse nulle est rejetée (p < alpha),
        et True si l'hypothèse nulle n'est pas rejetée (p >= alpha).
        La chaîne de caractères fournit une interprétation détaillée du résultat du test.
    """
    if p < alpha:
        text = f"[pval]p-value = {p}[/]\n" + "p < alpha --> Rejet de H0,\nacceptation de H1:\n" + H1
        return False, text
    else:
        text = f"[pval]p-value = {p}[/]\n" + "p > alpha --> Non rejet de H0,\nacceptation de H0 avec un risque beta:\n" + H0
        return True, text

def test_norm(echantillon, test: Literal["auto", "shapiro", "KS"]="auto", name_distrib=True,
                  alpha=0.05, rtn: Literal['bool&print', 'txt', 'bool', 'print']='bool&print'):
    """Teste si un échantillon suit une distribution normale en utilisant le test de Shapiro-Wilk ou le test de Kolmogorov-Smirnov.
    Parameters:
    echantillon (pd.Series): L'échantillon de données à tester.
    test (Literal["auto", "shapiro", "KS"], optional): Le test à utiliser. Par défaut "auto".
        - "auto": Utilise le test de Shapiro-Wilk pour les échantillons de taille <= 50, sinon le test de Kolmogorov-Smirnov.
        - "shapiro": Utilise toujours le test de Shapiro-Wilk.
        - "KS": Utilise toujours le test de Kolmogorov-Smirnov.
    name_distrib (bool, optional): Si True, inclut le nom de l'échantillon dans les hypothèses H0 et H1. Par défaut True.
    alpha (float, optional): Le niveau de signification pour le test. Par défaut 0.05.
    rtn (Literal['bool&print', 'txt', 'bool', 'print'], optional): Le format de retour. Par défaut 'bool&print'.
        - 'bool&print': Retourne un booléen et imprime l'interprétation.
        - 'txt': Retourne le nom du test et l'interprétation sous forme de texte.
        - 'bool': Retourne un booléen indiquant si H0 est acceptée.
        - 'print': Imprime l'interprétation.
    Returns:
    bool or tuple: Dépend de la valeur de `rtn`.
        - 'bool&print': Retourne un booléen indiquant si H0 est acceptée et imprime l'interprétation.
        - 'txt': Retourne un tuple (nom du test, interprétation).
        - 'bool': Retourne un booléen indiquant si H0 est acceptée.
        - 'print': Ne retourne rien, imprime seulement l'interprétation.
                """
    if name_distrib:
        H0 = f"[h]H0: La distribution {echantillon.name} [/][motclef]suit une loi normal[/]"
        H1 = f"[h]H1: La distribution {echantillon.name} [/][motclef]ne suit pas une loi normal[/]"
    else:
        H0 = f"[h]H0: La distribution [/][motclef]suit une loi normal[/]"
        H1 = f"[h]H1: La distribution [/][motclef]ne suit pas une loi normal[/]"
    
    p = None
    ### Test ###
    test_name=""
    # Le test de Shapiro est particulièrement puissant pour les petits n
    if (echantillon.count() <= 50 and test=="auto") or test=="shapiro":
        p = st.shapiro(echantillon).pvalue
        test_name = "shapiro"
    elif (echantillon.count() > 50 and test=="auto") or test=="KS":
        p = st.kstest(echantillon, "norm").pvalue
        test_name = "Kolmogorov-Smirnov"
    ### Interprétation ###
    interpretation = _interpretation_test(p, alpha, H0, H1)
    match rtn:
        case 'txt':
            return test_name, interpretation[1]
        case 'bool':
            return interpretation[0] 
        case 'print':
            consl.print(f"[test]Test de normalitée[/] de '{echantillon.name}' par [test]{test_name}[/]:\n",
                    interpretation[1], 
                    sep="\n")
        case 'bool&print':
            consl.print(f"[test]Test de normalitée[/] de '{echantillon.name}' par [test]{test_name}[/]:\n",
                    interpretation[1], 
                    sep="\n")
            return interpretation[0]

#### outil d'editon html ####
def _edit_table_html(axe_x, axe_y, table_np, totaux=True):
    # Paramètres de style gérer par balise css
    html_text_style = """
    <div style="display:inline">
        <style>
            td, th {
            padding: 5px;
        }
        table {
            font-size: .8rem;
        }
        .entete {
            background-color: #464646;
        }
        .xentete {
            background-color: #464646;
            text-align: left;  
        }
        .totalparciel {
            background-color: #151515;
            }
        </style>
        """
    html_table = "<table class='dataframe'>" + "<thead>" + "<tr>"
    for cel_value in axe_y:
        html_table += "<th class='entete'>" + str(cel_value).replace('(', ']').replace(')', ']') + "</th>"
    html_table += "</tr>" + "</thead>"
    for index_linge in range(table_np.shape[0]):
        html_table += "<tr>"
        html_table += "<th class='xentete'><strong>" + str(axe_x[index_linge]).replace('(', ']').replace(')', ']') + "</strong></th>"
        if index_linge < table_np.shape[0]-1:
            for index_cols in range(table_np.shape[1]):
                if index_cols < table_np.shape[1]-1:
                    html_table += "<td>" + str(table_np[index_linge, index_cols]) + "</td>"
                else:
                    html_table += "<td "
                    if totaux:
                        html_table += "class='totalparciel'><b>"
                    else:
                        html_table += ">"
                    html_table += str(table_np[index_linge, index_cols])
                    if totaux:
                        html_table += "</b>"
                    html_table += "</td>"
        else:
            for index_cols in range(table_np.shape[1]):
                html_table += "<td "
                if totaux:
                    html_table += "class='totalparciel'><b>"
                else:
                    html_table += ">"
                html_table += str(table_np[index_linge, index_cols])
                if totaux:
                        html_table += "</b>"
                html_table += "</td>"
        html_table += "</tr>"
    html_table += "</tbody></table></div>"
    html_table = html_text_style + html_table
    return html_table

def juxtapose_html(contenues_html: list[str], names_contenues: list[str], importance_titre=3):
    """Juxtapose plusieurs contenus HTML côte à côte avec des titres.

    Args:
        contenues_html (list[str]): Une liste de chaînes de contenu HTML à afficher.
        names_contenues (list[str]): Une liste de titres correspondant à chaque contenu HTML.
        importance_titre (int, optionnel): La taille des balises de titre HTML (h1, h2, etc.). Par défaut 3.

    Returns:
        None: Cette fonction affiche directement le contenu HTML et ne retourne rien.
    """
    html_str='<div style="display:inline-flex">'
    for contenue_html, title in zip(contenues_html, names_contenues): 
        html_str += '<div style="padding:10px" style="display:inline">'
        html_str += f'<h{importance_titre}>{title}</h{importance_titre}>'
        html_str += str(contenue_html)
        html_str += '</div>'
    html_str+='</div>' 
    display_html(html_str,raw=True)

class BaseBivariateOutil:
    def __init__(self, datframe: pd.DataFrame, colonne_x: Optional[str]=None, 
                 colonne_y: Optional[str]=None,
                 x_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 y_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 col_preformat_x: Optional[list]=None, col_preformat_y: Optional[list]=None,
                 nb_sep_QC = 10, nb_lim_QD = 10,
                 sep_round_x=0, sep_round_y=0):
        #"QuantiContinu" | "QuantiDiscret" | "QualiNominal" | "QualiOrdinal"
        # Récupération des colonnes
        self.df = datframe
        self.x = datframe.loc[:, colonne_x]
        self.y = datframe.loc[:, colonne_y]
        self.nb_sep_QC = nb_sep_QC
        self.nb_lim_QD = nb_lim_QD
        
        # Récupération des types de vartiables, si nn renseigner -> test
        if x_type:
            self.x_type = x_type
        else:
            self.x_type = self._test_type(self.x)
        if y_type:
            self.y_type = y_type
        else:
            self.y_type = self._test_type(self.y)
        
        # Vérification présence d'order si QualiOrdinal
        if self.x_type == "QualiOrdinal":
            if not col_preformat_x:
                raise ValueError("Veuillez renseigner col_preformat_x, pour les variables QualiOrdinal")
        if self.y_type == "QualiOrdinal":
            if not col_preformat_y:
                raise ValueError("Veuillez renseigner col_preformat_y, pour les variables QualiOrdinal")
        
        # crtéation des colonnes grouper
        self.col_tab_x = self._set_colonne(self.x, self.x_type, self.nb_sep_QC, sep_round_x, col_preformat_x)
        self.col_tab_y = self._set_colonne(self.y, self.y_type, self.nb_sep_QC, sep_round_y, col_preformat_y)
        self.table_np, self.axe_y, self.axe_x = self._set_np_table()
        
        # Table de fréquances:
        self.table_freq = self._np_freq()
    
    ### Partie initalisation ###
    # Fonction d'attribution automatique du type de variables
    def _test_type(self, serie):
        # Attribution de type de variables
        if ptypes.is_float_dtype(serie):
            return "QuantiContinu"
        elif ptypes.is_integer_dtype(serie):
            if len(serie.unique()) > self.nb_lim_QD:
                return "QuantiContinu"
            return "QuantiDiscret"
        elif (ptypes.is_string_dtype(serie)) or (ptypes.is_bool_dtype(serie)):
            return "QualiNominal"
        else:
            return "Uncknown"
    
    def _set_colonne(self, col, type_col, sep, sep_round, col_preformat):
        col_tab = []
        if col_preformat is not None:
            return col_preformat
        elif type_col == "QuantiContinu":
            # Détermine le min et le max arrondi de façon à englober la valeur exacte
            min_value = floor(col.min()) - 1
            max_value = ceil(col.max()) + 1
            # Calcul des séparateures en fontion de la présision demander
            if not sep_round:
                separateurs = np.linspace(min_value, max_value, (sep+1)).astype(int)
            else:
                separateurs = np.linspace(min_value, max_value, (sep+1)).round(decimals=sep_round)
            # Transformation des séparateures en intervales: un tuple tel que (borne inf, borne sup)
            separateurs = list(zip(separateurs, separateurs[1:]))
            col_tab= separateurs
            return col_tab
        elif type_col == "QuantiDiscret":
            col_tab = col.unique()
            col_tab.sort()
            return col_tab
        elif type_col == "QualiNominal":
            col_tab = col.unique()
            try:
                col_tab.sort_values(inplace=True)
            except AttributeError:
                col_tab.sort()
            return col_tab
    
    def _set_np_table(self):
        # Axe y (entètes)
        line_top_y = np.concatenate(
            (np.array(["X\Y"], ndmin=2),
             np.array(self._item_to_str(self.col_tab_y), ndmin=2),
             np.array(["Total_X"], ndmin=2)), 1)
        # Axe x (entètes, lignes)
        line_left_x = np.concatenate(
            (np.array(self._item_to_str(self.col_tab_x),
                      ndmin=2), np.array(["Total_Y"], ndmin=2)), 1)
        # Corps de la table
        line_top_temp = np.zeros((1, len(self.col_tab_y)+1), dtype=int)
        for x_value in self.col_tab_x:
            line_x_valiue = np.array([], ndmin=2, dtype=int)
            for y_value in self.col_tab_y:
                valiue_case = self._count_nxy(x_value, y_value)
                line_x_valiue = np.concatenate((line_x_valiue, np.array([valiue_case], ndmin=2, dtype=int)), 1, dtype=int)
            line_x = np.concatenate((line_x_valiue, np.array([line_x_valiue.sum()], ndmin=2, dtype=int)), 1, dtype=int)
            line_top_temp = np.concatenate((line_top_temp, line_x), dtype=int)
        line_btm = line_top_temp.sum(0).reshape((1, len(self.col_tab_y)+1))
        table_entete_less = np.concatenate((line_top_temp, line_btm), 0, dtype=int)
        return table_entete_less[1:,:], line_top_y[0], line_left_x[0]

    def _count_nxy(self, x_value, y_value):
        query = []
        for colonne_value, type_val in (((self.x, x_value), self.x_type), ((self.y, y_value), self.y_type)):
            clo_name = colonne_value[0].name
            value = colonne_value[1]
            if type_val == "QuantiContinu":
                query.append(f"{value[1]} >= `{clo_name}` > {value[0]}")
            if type_val== "QuantiDiscret":
                if type(value) == str:
                    if value[-1] == "-":
                        value_rec = float(value.split(" ")[0])
                        value_min = colonne_value[0].min()
                        query.append(f"{value_rec} >= `{clo_name}` >= {value_min}")
                    elif value[-1] == "+":
                        value_rec = float(value.split(" ")[0])
                        value_max = colonne_value[0].max()
                        query.append(f"{value_max} >= `{clo_name}` >= {value_rec}")
                elif type(value) == list or type(value) == tuple:
                    query.append(f"{value[1]} >= `{clo_name}` > {value[0]}")
                else:
                    query.append(f"`{clo_name}` == {value}")
            if type_val== "QualiNominal":
                query.append(f"`{clo_name}` == '{value}'")
        query = " & ".join(query)
        
        count = self.df.query(query).loc[:, self.x.name].count()
        return count

    def _item_to_str(self, colonne):
        col_tab = []
        for item in colonne:
            col_tab.append(f"{item}")
        return col_tab

    def _np_freq(self):
        total = self.table_np[-1,-1]
        table_freq = self.table_np / total
        return table_freq

    def set_axes(self, axex=None, axey=None, echange_axes=False):
        """
        Définit les axes pour l'objet courant.
        Paramètres:
        -----------
        axe(x|y) : List, optionnel
            - Liste contenant les valeurs des axes x ou y:
                - Pour les variables 'Quanti', les valeurs doivent être des tuples. (borne inf, borne sup)
                - Pour les variables 'Quali', les valeurs doivent être des str.
        echange_axes : bool, optionnel
            Si True, échange les axes x et y ainsi que leurs types respectifs.
        Actions:
        --------
        - Met à jour les colonnes utilisées pour les axes x et y.
        - Échange les axes x et y si `echange_axes` est True.
        """
        if axex:
            self.col_tab_x=axex
        if axey:
            self.col_tab_y=axey
        if echange_axes:
            self.col_tab_x, self.col_tab_y = self.col_tab_y, self.col_tab_x
            self.x, self.y = self.y, self.x
            self.x_type, self.y_type = self.y_type, self.x_type
        
        # Table de contingence:
        self.table_np, self.axe_y, self.axe_x = self._set_np_table()
        
        # Table de fréquances:
        self.table_freq = self._np_freq()
    
class TableauContingence(BaseBivariateOutil):
    def __init__(self, datframe: pd.DataFrame, colonne_x: Optional[str]=None, 
                 colonne_y: Optional[str]=None,
                 x_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 y_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 col_preformat_x: list=None, col_preformat_y: list=None,
                 nb_sep_QC = 10, nb_lim_QD = 10,
                 sep_round_x=0, sep_round_y=0):
        #"QuantiContinu" | "QuantiDiscret" | "QualiNominal" | "QualiOrdinal"
        # Récupération des colonnes
        BaseBivariateOutil.__init__(self, datframe, 
                                    colonne_x, colonne_y, 
                                    x_type, y_type, 
                                    col_preformat_x, col_preformat_y, 
                                    nb_sep_QC, nb_lim_QD, 
                                    sep_round_x, sep_round_y)
        
        # Formatage de la table html afficher par l'objet
        self.text_html = _edit_table_html(self.axe_x, self.axe_y, self.table_np)
    
    #### Partie affichage ####
    def _repr_html_(self):
        return self.text_html
    
    ### Partie tables ###    
    def set_axes(self, axex=None, axey=None, echange_axes=False):
        """
        Définit les axes pour l'objet courant.
        Paramètres:
        -----------
        axe(x|y) : List, optionnel
            - Liste contenant les valeurs des axes x ou y:
                - Pour les variables 'Quanti', les valeurs doivent être des tuples. (borne inf, borne sup)
                - Pour les variables 'Quali', les valeurs doivent être des str.
        echange_axes : bool, optionnel
            Si True, échange les axes x et y ainsi que leurs types respectifs.
        Actions:
        --------
        - Met à jour les colonnes utilisées pour les axes x et y.
        - Échange les axes x et y si `echange_axes` est True.
        - Met à jour la table de contingence, la table HTML et la table de fréquences en fonction des nouveaux axes.
        """
        super().set_axes(axex, axey, echange_axes)
        
        # Formatage de la table html afficher par l'objet
        self.text_html = _edit_table_html(self.axe_x, self.axe_y, self.table_np)
    
    def frequences(self, pourcentages=False, retun_text=False):
        table_freq = self.table_freq.copy()
        if pourcentages:
            table_freq *= 100
            table_freq = table_freq.round(2)
        else:
            table_freq = table_freq.round(4)            
        table_freq_html = _edit_table_html(self.axe_x, self.axe_y, table_freq)
        if retun_text:
            return table_freq_html
        else:
            return HTML(table_freq_html)

    def frequances_partiels(self, axe: Literal["h", "v", "hv", "vh"]="hv", html=True, pct=True):
        frequances_partiels_h = None
        frequances_partiels_v = None
        if "h" in axe:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", category=RuntimeWarning) 
                frequances_partiels_h = np.divide(self.table_np, self.table_np[:, -1].reshape(self.table_np.shape[0], 1)).round(4)
            if pct: frequances_partiels_h = (frequances_partiels_h*100).round(2)
        if "v" in axe:  
            frequances_partiels_v = np.divide(self.table_np, self.table_np[-1,:]).round(4)
            if pct: frequances_partiels_v = (frequances_partiels_v*100).round(2)
        if frequances_partiels_h is not None and frequances_partiels_v is not None:
            if html:
                frequances_partiels_h = _edit_table_html(self.axe_x, self.axe_y, frequances_partiels_h)
                frequances_partiels_v = _edit_table_html(self.axe_x, self.axe_y, frequances_partiels_v)
                juxtapose_html(contenues_html=[frequances_partiels_h, frequances_partiels_v], 
                               names_contenues=["Fréquences partielles horizontales", "Fréquences partielles verticales"])
                return 
            else:
                return frequances_partiels_h, frequances_partiels_v
        elif frequances_partiels_h is not None:
            if html:
                return HTML(frequances_partiels_h)
            else:
                return frequances_partiels_h
        elif frequances_partiels_v is not None:
            if html:
                return HTML(frequances_partiels_v)
            else:
                return frequances_partiels_v
        else:
            raise Exception("ATTENTION: Veuillez renseigner un axe valide: 'h', 'v', 'hv' ou 'vh'")
    
    def contingence_et_frequences(self, importance_titre=3, pourcentages=True):
        juxtapose_html(
            contenues_html= [self.text_html, 
                             self.frequences(pourcentages=pourcentages, retun_text=True)], 
            names_contenues= ["Table de contingence", "Table des fréquences (%)"], 
            importance_titre=importance_titre
            )

class TestStatBivar(BaseBivariateOutil):
    
    consl = Console(
        theme=Theme({
        "test": "#FBCE9E",
        "var": "#00BDC8 bold",
        "pval": "#00BDC8 bold",
        "h": "#7ACFB0",
        "chemin": "#F88F52 bold",
        "motclef": "#7ACFB0 underline",
        "choix": "#7ACFB0 underline bold",
        "espace": "#1C5588 bold"
        }, inherit=False), width=70)
    
    def __init__(self, 
                 data: Union[pd.DataFrame, TableauContingence], 
                 colonne_x: Optional[str]=None, 
                 colonne_y: Optional[str]=None,
                 x_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 y_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 col_preformat_x: list=None, col_preformat_y: list=None,
                 nb_sep_QC = 10, nb_lim_QD = 10,
                 sep_round_x=0, sep_round_y=0,
                 echantillon_apparies=None, lim_affiche_text = 15):
        
        if isinstance(data, pd.DataFrame):
            # Si l'entrée est un DataFrame, il faut obligatoirement fournir les noms de séries
            if colonne_x is None or colonne_y is None:
                raise ValueError("Pour un DataFrame, colonne_x et colonne_y doivent être spécifiés.")
            BaseBivariateOutil.__init__(self, data, 
                                        colonne_x, colonne_y, 
                                        x_type, y_type, 
                                        col_preformat_x, col_preformat_y, 
                                        nb_sep_QC, nb_lim_QD, 
                                        sep_round_x, sep_round_y)
            
        elif isinstance(data, TableauContingence):
            # Dans le cas d'un objet TableauContingence, on récupère directement les variables
            BaseBivariateOutil.__init__(self, data.df, 
                                        data.x.name, data.y.name, 
                                        data.x_type, data.y_type, 
                                        data.col_tab_x, data.col_tab_y, 
                                        data.nb_sep_QC, data.nb_lim_QD)
        else:
            raise TypeError("L'argument data doit être un DataFrame Pandas ou un TableauContingence.")
        
        self.lim_affiche_text = lim_affiche_text
        self.apparies = echantillon_apparies

    #### Partie stat ####
    def _quali_Quanti(self):
        # iddentifie les types de variables
        patern = r"Q.*[C|D|N|O]"
        variablex_type, variabley_type = re.search(patern, self.x_type).group()[:-1], re.search(patern, self.y_type).group()[:-1]
        # Et les retourne dans l'ordre quali puis quanti
        if variablex_type == "Quali":
            variable_quali = self.x
            variable_quanti = self.y
            return variable_quali, variable_quanti
        elif variabley_type == "Quali":
            variable_quali = self.y
            variable_quanti = self.x
            return variable_quali, variable_quanti
        else:
            consl.print("ATTENTION: nécessite au moins une variable Qualitative")
    
    #### Interprétation des tests ####
    def _test_d_intencitee(self,value):
        value = abs(value)
        interpretation = ""
        if value > 0.9:
            interpretation += "et sont [motclef]très fortement corréler[/]"
        elif value > 0.75:
            interpretation += "et sont [motclef]fortement corréler[/]"
        elif value > 0.60:
            interpretation += "et sont [motclef]assez fortement corréler[/]"
        elif value > 0.40:
            interpretation += "et sont [motclef]modérément corréler[/]"
        elif value > 0.20:
            interpretation += "et sont [motclef]peut corréler[/]"
        elif value > 0.05:
            interpretation += "et sont [motclef]très peut corréler[/]"
        else:
            interpretation += "mais de façon [motclef]quasi-négligable[/]"
        return interpretation
    
    def _interpretation_cov(self, value, intencite=False):
        interpretation = ""
        x_name = self.x.name if len(self.x.name) < self.lim_affiche_text else self.x.name[:self.lim_affiche_text] + "..."
        y_name = self.y.name if len(self.y.name) < self.lim_affiche_text else self.y.name[:self.lim_affiche_text] + "..."
        if value < 0.0:
            interpretation = f"'{x_name}' et '{y_name}' évoluent dans des [motclef]directions opposées[/]"
            if intencite:
                interpretation += " " + self._test_d_intencitee(value)
        elif value > 0.0:
            interpretation = f"'{x_name}' et '{y_name}' évoluent dans le [motclef]même sens[/]"
            if intencite:
                interpretation += " " + self._test_d_intencitee(value)
        else:
            interpretation = f"'{x_name}' et '{y_name}' évoluent [motclef]indépendament l'un de l'autre[/]"
        return interpretation
    
    #### test de vérification des conditon de validité ####
    def test_para(self, echantillon, **kwars):
        # Pour savoir si l'on peut utiliser les méthodes paramétriques 
        # on fait un test de normalité
        return test_norm(echantillon, **kwars)
    
    def test_egalite_variances(self, alpha=0.01):
        quali, Quanti = self._quali_Quanti()
        H0 = f"[h]H0: Les variances des groupes [/][motclef]sont égales[/]"
        H1 = f"[h]H1: Les variances des groupes [/][motclef]ne sont pas égales[/]"
        ### Test ###
        liste_de_serie = []
        for categ in quali.unique():
            liste_de_serie.append(self.df.loc[self.df.groupby(by=f"{quali.name}", observed=False).groups[categ], f"{Quanti.name}"])
        p = st.bartlett(*liste_de_serie).pvalue
        ### Interprétation ###
        interpretation = _interpretation_test(p, alpha, H0, H1)
        consl.print("[test]Test d'égalitées des variances de Bartlett[/]:\n",
                    interpretation[1],
                    sep="\n")
        return interpretation[0]
    
    #### test bivariées ####
    def covariance(self, data_forme=False):
        value = self.x.cov(self.y)
        interpretation = self._interpretation_cov(value)
        if data_forme:
            return round(value, 2) ,interpretation
        consl.print(f"Covariance = {round(value, 3)}",
                    interpretation, 
                    sep="\n")
    
    def correlation_pearson(self, data_forme=False):
        value, p = st.pearsonr(self.x, self.y)
        value_carre = value*value
        interpretation = self._interpretation_cov(value, intencite=True)
        x_name = self.x.name if len(self.x.name) < self.lim_affiche_text else self.x.name[:self.lim_affiche_text] + "..."
        y_name = self.y.name if len(self.y.name) < self.lim_affiche_text else self.y.name[:self.lim_affiche_text] + "..."
        if data_forme:
            return value, p ,interpretation
        consl.print(f"[test]Coefficient de corrélation de Pearson[/] ([pval]p={p}[/]): ",
              f"[pval]r = {round(value, 3)}[/]",
              interpretation, "",
              "[test]Coefficient de détermination[/]:",
              f"[pval]r² = {round(value_carre, 3)}[/]",
              f"la variable [var]{x_name}[/] explique à [motclef]{round(value_carre*100, 1)}%[/] la variable [var]{y_name}[/]"
              , sep="\n")
    
    def correlation_spearman(self, data_forme=False):
        value, p = st.spearmanr(self.x, self.y)
        value_carre = value*value
        interpretation = self._interpretation_cov(value, intencite=True)
        x_name = self.x.name if len(self.x.name) < self.lim_affiche_text else self.x.name[:self.lim_affiche_text] + "..."
        y_name = self.y.name if len(self.y.name) < self.lim_affiche_text else self.y.name[:self.lim_affiche_text] + "..."
        if data_forme:
            return value ,interpretation
        consl.print(f"[test]Coefficient de corrélation Spearman[/] ([pval]p={p}[/]): ",
              f"[pval]r = {round(value, 3)}[/]",
              interpretation, "",
              "[test]Coefficient de détermination[/]:",
              f"[pval]r² = {round(value_carre, 3)}[/]",
              f"la variable [var]{x_name}[/] explique à [motclef]{round(value_carre*100, 1)}%[/] la variable [var]{y_name}[/]"
              , sep="\n")
    
    def test_independance_khi_deux(self, alpha=0.01, data_forme=False):
        # Test
        stat_khi, p, ddl, ex = st.chi2_contingency(self.table_np[:-1,:-1])
        x_name = self.x.name if len(self.x.name) < self.lim_affiche_text else self.x.name[:self.lim_affiche_text] + "..."
        y_name = self.y.name if len(self.y.name) < self.lim_affiche_text else self.y.name[:self.lim_affiche_text] + "..."
        # Texte d'introduction
        H0 = f"[h]H0: les variables {x_name} et {y_name} [/][motclef]sont indépendantes[/]"
        H1 = f"[h]H1: les variables {x_name} et {y_name} [/][motclef]ne sont pas indépendantes[/]"
        # Print des résultats
        if data_forme:
            return stat_khi, _interpretation_test(p, alpha, H0, H1)[1]
        consl.print(f"[test]Test X² d'indépendance des variables[/] {x_name} et {y_name}:\n",
                    f"[pval]X² = {stat_khi}[/]",
                    f"[pval]ddl = {ddl}[/]\n",
                    ### Interprétation ###
                    _interpretation_test(p, alpha, H0, H1)[1],
                    sep="\n")
        
        ### Affichage des tables ###
        tb_initial = _edit_table_html(self.axe_x[:-1], self.axe_y[:-1], self.table_np[:-1,:-1], totaux=False)
        tb_attendue = _edit_table_html(self.axe_x[:-1], self.axe_y[:-1], ex, totaux=False)
        juxtapose_html(contenues_html=[tb_initial, tb_attendue], 
                       names_contenues=["Table initial", "Table attendu"], 
                       importance_titre=4)
    
    def test_kruskal_wallis(self, alpha=0.01, data_forme=False):
        # Iddentifie les var quali et quanti
        quali, quanti = self._quali_Quanti()
        quali_name = quali.name if len(quali.name) < self.lim_affiche_text else quali.name[:self.lim_affiche_text] + "..."
        quanti_name = quanti.name if len(quanti.name) < self.lim_affiche_text else quanti.name[:self.lim_affiche_text] + "..."
        # Mise en liste des series par catégories puis passage dans la fontion
        liste_de_serie = []
        for categ in quali.unique():
            liste_de_serie.append(self.df.loc[self.df.groupby(by=f"{quali.name}", observed=False).groups[categ], f"{quanti.name}"])
        stat_krus, p = st.kruskal(*liste_de_serie)
        # Texte d'introduction 
        H0 = f"[h]H0: les médianes des catégories de {quali_name} en fontion des {quanti_name} [/][motclef]sont égales[/]"
        H1 = f"[h]H1: les médianes des catégories de {quali_name} en fontion des {quanti_name} [/][motclef]ne sont pas égales[/]"
        # Print des résultats
        if data_forme:
            return stat_krus, _interpretation_test(p, alpha, H0, H1)[1]
        consl.print(f"[test]Test de Kruskal-Wallis d'égalité des médianes[/] sur {quali_name} et {quanti_name}:\n",
                    # consl.Print des résultats du test 
                    f"stat: [pval]H = {stat_krus}[/]",
                    f"[pval]p-value = {p}[/]\n",
                    ### Interprétation ###
                    _interpretation_test(p, alpha, H0, H1)[1],
                    sep="\n")
        
    # rapport des interactions entre les variables par le test statistique
    def analyse_bivar_et_test(self, apparies=None, rtn_txt=False):    #anciènement: test_adequat
        # Permet de renseigner l'appariment ou non au niveau de la fonction
        if apparies == None:
            apparies = self.apparies
        self.apparies = apparies
        table_attendue = st.contingency.expected_freq(self.table_np[:-1, :-1])
        # Iddentifications des tyypes de variables
        patern = r"Q.*[C|D|N|O]"
        variablex_type, variabley_type = re.search(patern, self.x_type).group()[:-1], re.search(patern, self.y_type).group()[:-1]
        rtn='bool'
        # Cas de l'utilisation dans la console
        if not rtn_txt:
            rtn='bool&print'
            consl.print(f"[var]{self.x.name}[/] (x): [motclef]{variablex_type}tative[/]  |  [var]{self.y.name}[/] (y): [motclef]{variabley_type}tative[/]")
        
        ## Edition de l'arbre de désisions ##
        trace_des_choix = ""
        # Cas de deux variables Quantitatives
        if variablex_type == "Quanti" and variabley_type == "Quanti":
            trace_des_choix += "[chemin]2 Quanti[/] --> "
            # Test si les variables sont paramétriques
            if self.test_para(self.x, rtn=rtn) and self.test_para(self.y, rtn=rtn):
                if rtn_txt:
                    return "correlation_pearson", self.correlation_pearson(data_forme=True)
                trace_des_choix += "[chemin]2 paramétriques[/]:"
                consl.print(trace_des_choix, 
                            "[choix]correlation_pearson[/]:",
                            sep="\n", overflow='fold')
                self.correlation_pearson()
            else:
                if rtn_txt:
                    return "correlation_spearman", self.correlation_spearman(data_forme=True)
                trace_des_choix += "[chemin]>= 1 non paramétrique[/]:"
                consl.print(trace_des_choix,
                            "[choix]correlation_spearman[/]:",
                            sep="\n", overflow='fold')
                self.correlation_spearman()
        
        # Cas une variable Quantitative et une Qualitative
        elif variablex_type == "Quali" and variabley_type == "Quanti" or variablex_type == "Quanti" and variabley_type == "Quali":
            trace_des_choix += "[chemin]1 Quanti & 1 Quali[/] --> "
            variable_quali, variable_quanti = self._quali_Quanti()
            if variable_quali.nunique() > 2:
                trace_des_choix += "[chemin]Quali > 2 groupes[/] --> "
                if apparies == None:
                    consl.print("Veuillez renseigner si les échantillons sont appariées")
                elif apparies:
                    trace_des_choix += "[chemin]Échantillons appariés[/] --> "
                    if self.test_para(variable_quanti, rtn=rtn):
                        if rtn_txt:
                            return "mesures_anova_repeter"                                ####  mesures_anova_repeter à implémenter
                        trace_des_choix += "[chemin]Quanti paramétrique[/]:"
                        consl.print(trace_des_choix,
                                    "[choix]mesures_anova_repeter[/]:",
                                    sep="\n", overflow='fold')
                    else:
                        if rtn_txt:
                            return "test_friedman"                                        ####  test_friedman à implémenter
                        trace_des_choix += "[chemin]Quanti non paramétrique[/]:"
                        consl.print(trace_des_choix,
                                    "[choix]test_friedman[/]:",
                                    sep="\n", overflow='fold')
                else:
                    trace_des_choix += "[chemin]Échantillons indépendants[/] --> "
                    if self.test_para(variable_quanti, rtn=rtn):
                        trace_des_choix += "[chemin]Quanti paramétrique[/] --> "
                        if self.test_egalite_variances():
                            if rtn_txt:
                                return "test_anova_unidirectionel"                        ####  test_anova_unidirectionel à implémenter
                            trace_des_choix += "[chemin]Variances égales[/]:"
                            consl.print(trace_des_choix,
                                        "[choix]test_anova_unidirectionel[/]:",
                                        sep="\n", overflow='fold')
                        else:
                            if rtn_txt:
                                return "test_anova_welch"                                 ####  test_anova_welch à implémenter
                            trace_des_choix += "[chemin]Variances inégales[/]:"
                            consl.print(trace_des_choix,
                                        "[choix]test_anova_welch[/]:",
                                        sep="\n", overflow='fold')
                    else:
                        if rtn_txt:
                            return "test_kruskal_wallis", self.test_kruskal_wallis(data_forme=True)
                        trace_des_choix += "[chemin]Quanti non paramétrique[/]:"
                        consl.print(trace_des_choix,
                                    "[choix]test_kruskal_wallis[/]:",
                                    sep="\n", overflow='fold')
                        self.test_kruskal_wallis()
            else:
                trace_des_choix += "[chemin]Quali = 2 groupes[/] --> "
                if apparies == None:
                    consl.print("Veuillez renseigner si les échantillons sont appariées")
                elif apparies:
                    trace_des_choix += "[chemin]Échantillons appariés[/] --> "
                    if self.test_para(variable_quanti, rtn=rtn):
                        if rtn_txt:
                            return "test_t_student_pour_echantillons_apparies"             ####  test_t_student_pour_echantillons_apparies à implémenter
                        trace_des_choix += "[chemin]Quanti paramétrique[/]:"
                        consl.print(trace_des_choix,
                                    "[choix]test_t_student_pour_echantillons_apparies[/]:",
                                    sep="\n", overflow='fold')
                    else:
                        if rtn_txt:
                            return "test_rangs_signes_Wilcoxon"                            ####  test_rangs_signes_Wilcoxon à implémenter
                        trace_des_choix += "[chemin]Quanti non paramétrique[/]:"
                        consl.print(trace_des_choix,
                                    "[choix]test_rangs_signes_Wilcoxon[/]:",
                                    sep="\n", overflow='fold')
                else:
                    trace_des_choix += "[chemin]Échantillons indépendants[/] --> "
                    if self.test_para(variable_quanti, rtn=rtn):
                        trace_des_choix += "[chemin]Quanti paramétrique[/] --> "
                        if self.test_egalite_variances():
                            if rtn_txt:
                                return "test_t_student_pour_echantillons_inde"               ####  test_t_student_pour_echantillons_inde à implémenter
                            trace_des_choix += "[chemin]Variances inégales[/]:"
                            consl.print(trace_des_choix,
                                        "[choix]test_t_student_pour_echantillons_inde[/]:",
                                        sep="\n", overflow='fold')
                        else:
                            if rtn_txt:
                                return "test_t_welch_pour_echantillons_inde"                  ####  test_t_welch_pour_echantillons_inde à implémenter
                            trace_des_choix += "[chemin]Variances inégales[/]:"
                            consl.print(trace_des_choix,
                                        "[choix]test_t_welch_pour_echantillons_inde[/]:",
                                        sep="\n", overflow='fold')
                    else:
                        if rtn_txt:
                            return "test_mann_whitney_u"                                      ####  test_mann_whitney_u à implémenter
                        trace_des_choix += "[chemin]Quanti non paramétrique[/]:"
                        consl.print(trace_des_choix,
                                    "[choix]test_mann_whitney_u[/]:",
                                    sep="\n", overflow='fold')
        
        # Cas de deux variables Qualitatives
        elif variablex_type == "Quali" and variabley_type == "Quali":
            trace_des_choix += "[chemin]2 Quali[/] --> "
            # Test si plus de 2 groupes dans au mois une variable
            if (self.x.nunique() > 2 or self.y.nunique() > 2) and ((table_attendue >= 5).any() and (self.table_np[:-1, :-1] >= 5).any()):
                if rtn_txt:
                    return "test_independance_khi_deux", self.test_independance_khi_deux(data_forme=True)
                trace_des_choix += "[chemin]+2 groupes par var[/] et [chemin]les Tables n'ont pas de n < 5[/]:"
                consl.print(trace_des_choix,
                            "[choix]test_independance_khi_deux[/]:",
                            sep="\n", overflow='fold')
                self.test_independance_khi_deux()
            else:
                trace_des_choix += "[chemin]2 groupes par var[/] --> "
                if apparies == None:
                    consl.print("Veuillez renseigner si les variables sont appariées")
                elif apparies:
                    if rtn_txt:
                        return "test_mcnemor"                                                 ####  test_mcnemor à implémenter
                    trace_des_choix += "[chemin]Échantillons appariés[/]:"
                    consl.print(trace_des_choix,
                                "[choix]test_mcnemor[/]:",
                                sep="\n", overflow='fold')
                else:
                    trace_des_choix += "[chemin]Échantillons non appariés[/] --> "
                    if (table_attendue < 5).any() and (self.table_np[:-1, :-1] < 5).any():
                        if rtn_txt:
                            return "test_exact_ficher"                                        ####  test_exact_ficher à implémenter
                        trace_des_choix += "[chemin]Tables n'ont pas de n < 5[/]:"
                        consl.print(trace_des_choix,
                                    "[choix]test_exact_ficher[/]:",
                                    sep="\n", overflow='fold')
                    else:
                        if rtn_txt:
                            return "test_independance_khi_deux", self.test_independance_khi_deux(data_forme=True)
                        trace_des_choix += "[chemin]Tables ont un/des n < 5[/]:"
                        consl.print(trace_des_choix,
                                    "[choix]test_independance_khi_deux[/]:",
                                    sep="\n", overflow='fold')
                        self.test_independance_khi_deux()
        
        else:
            consl.print("[Erreur]Erreur:[\] mauvaise identification des variables")
    
class StatBivarPlot(TestStatBivar, TableauContingence):
    
    def __inti__(self, 
                 data: Union[pd.DataFrame, TableauContingence], 
                 colonne_x: Optional[str]=None, 
                 colonne_y: Optional[str]=None,
                 x_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 y_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 col_preformat_x: list=None, col_preformat_y: list=None,
                 nb_sep_QC = 10, nb_lim_QD = 10,
                 sep_round_x=0, sep_round_y=0):
        
        if isinstance(data, pd.DataFrame):
            # Si l'entrée est un DataFrame, il faut obligatoirement fournir les noms de séries
            if colonne_x is None or colonne_y is None:
                raise ValueError("Pour un DataFrame, colonne_x et colonne_y doivent être spécifiés.")
            TestStatBivar.__init__(self, data, 
                                colonne_x, colonne_y, 
                                x_type, y_type, 
                                col_preformat_x, col_preformat_y, 
                                nb_sep_QC, nb_lim_QD, 
                                sep_round_x, sep_round_y)
            
        elif isinstance(data, TableauContingence):
            # Dans le cas d'un objet TableauContingence, on récupère directement les variables
            TestStatBivar.__init__(self, data.df, 
                                data.x.name, data.y.name, 
                                data.x_type, data.y_type, 
                                data.col_tab_x, data.col_tab_y, 
                                data.nb_sep_QC, data.nb_lim_QD)
        else:
            raise TypeError("L'argument data doit être un DataFrame Pandas ou un TableauContingence.")
    
    #### Partie représentation graphique avec seaborn ####
    # Représentations avancées sans ax renseignables
    def plot_rel(self, **kwargs):
        sns.relplot(data=self.df, x=self.x.name, y=self.y.name, **kwargs)

    def plot_lm(self, **kwargs):
        sns.lmplot(data=self.df, x=self.x.name, y=self.y.name, **kwargs)
        
    def plot_cat(self, **kwargs):
        sns.catplot(data=self.df, x=self.x.name, y=self.y.name, **kwargs)
    
    def plot_joint(self, **kwargs):
        sns.jointplot(data=self.df, x=self.x.name, y=self.y.name, **kwargs)
     
    # Représentations élémentaires avec ax renseignables
    def plot_box(self, ax=None, figsize=(5, 5), inverse=False, orient="v", **kwargs):
        if 'Quali' in self.y_type :
            categ = self.y.name
            conti = self.x.name
        elif 'Quali' in self.x_type :
            categ = self.x.name
            conti = self.y.name
        else:
            return 'Au moins une des catégories doit être qualitative'
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        
        if inverse or orient=="h":
            sns.boxplot(data=self.df.sort_values(categ), x=conti, y=categ, ax=ax, **kwargs) 
        else:
            sns.boxplot(data=self.df.sort_values(categ), x=categ, y=conti, ax=ax, **kwargs) 
    
    def plot_violin(self, ax=None, figsize=(5, 5), **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        sns.violinplot(data=self.df, x=self.x.name, y=self.y.name, ax=ax, **kwargs)
    
    def plot_scatter(self, ax=None, figsize=(5, 5), **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(data=self.df, x=self.x.name, y=self.y.name, ax=ax, **kwargs)
    
    def plot_line(self, ax=None, figsize=(5, 5), **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(data=self.df, x=self.x.name, y=self.y.name, ax=ax, **kwargs)
    
    def plot_heatmap(self, linewidths=.5, kind: Literal["ctg", "frec", "frec_h", "frec_v", "ctg_frec"]="ctg",
                pct=True, ax=None, xticklabels='auto', yticklabels='auto', figsize=(5, 5), **kwargs):
        if xticklabels == 'auto':
            xticklabels=self.axe_y[1:-1]
        if yticklabels == 'auto':
            yticklabels=self.axe_x[:-1]
        
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        if kind == "ctg":
            sns.heatmap(data=self.table_np[:-1,:-1], linewidths=linewidths, xticklabels=xticklabels,
                        yticklabels=yticklabels, ax=ax, **kwargs)
            ax.set_xlabel(f'{self.y.name}')
            ax.set_ylabel(f'{self.x.name}')
        elif kind == "frec":
            if pct:
                table = self.table_freq[:-1,:-1] * 100
            else:
                table = self.table_freq[:-1,:-1]
            sns.heatmap(data=table, linewidths=linewidths, xticklabels=xticklabels,
                        yticklabels=yticklabels, ax=ax, **kwargs)
            ax.set_xlabel(f'{self.y.name}')
            ax.set_ylabel(f'{self.x.name}')
        elif kind == "frec_h":
            sns.heatmap(data=self.frequances_partiels(axe="h", html=False, pct=pct)[:-1,:-1], linewidths=linewidths, xticklabels=xticklabels,
                        yticklabels=yticklabels, ax=ax, **kwargs)
            ax.set_xlabel(f'{self.y.name}')
            ax.set_ylabel(f'{self.x.name}')
        elif kind == "frec_v":
            sns.heatmap(data=self.frequances_partiels(axe="v", html=False, pct=pct)[:-1,:-1], linewidths=linewidths, xticklabels=xticklabels,
                        yticklabels=yticklabels, ax=ax, **kwargs)
            ax.set_xlabel(f'{self.y.name}')
            ax.set_ylabel(f'{self.x.name}')
        elif kind == "ctg_frec":
            map_pct = self.table_freq[:-1,:-1] * 100
            sns.heatmap(data=self.table_np[:-1,:-1], linewidths=linewidths, xticklabels=xticklabels,
                        yticklabels=yticklabels, ax=ax, annot=map_pct, **kwargs)
            ax.set_xlabel(f'{self.y.name}')
            ax.set_ylabel(f'{self.x.name}')
        else:
            print("ATTENTION: Veuillez renseigner un kind valide")
    
    # Résumer graphique:
    def plot_annalyse_biv(self, apparies=None, cbar_hauteur=0.8, ctg_heat_kind: Literal["ctg", "frec", "ctg_frec"]="ctg",
                     limite_char_label=10, lim_affiche_text=15, figure_droite: Literal["boxplot", 'boxplot_inv', "heatmap", "scatterplot"]= 'auto',
                     kwargs_boxplot=dict(), kwargs_heatmap_ctg=dict(), kwargs_heatmap_par=dict(), kwargs_heatmap_quali=dict(),
                     kwargs_scatterplot=dict(), kwargs_histplot=dict(), kwargs_countplot=dict()):
        
        self.lim_affiche_text = lim_affiche_text
        
        #### Création de la figure ####
        ### Figure ###
        fig = plt.figure(figsize=(22, 11))

        ### Grille ###
        global_grid_titre = gridspec.GridSpec(2, 1, figure = fig, height_ratios=(1, 14), hspace=0.15,)
        
        global_grid = global_grid_titre[1, 0].subgridspec(1, 2, width_ratios=(6, 5), wspace=0.16,)
        grid_gauche = global_grid[0, 0].subgridspec(3, 1, hspace=0.25, height_ratios=(20, 40, 1))
        grid_gauche_bas = grid_gauche[1, 0].subgridspec(1, 3, wspace=0.35, width_ratios=(1, 20, 30))
        grid_gauche_bas_droite = grid_gauche_bas[0, 2].subgridspec(3, 1, hspace=0.15, height_ratios=(10, 90, 5))
        grid_gauche_bas_gauche = grid_gauche_bas[0, 1].subgridspec(2, 1, hspace=0.3)
        
        if self.x_type != "QuantiContinu" and self.y_type != "QuantiContinu":
            grid_droite = global_grid[0, 1].subgridspec(2, 1, hspace=0.15, height_ratios=(5, 1.3))
            grid_droite_haut = grid_droite[0, 0].subgridspec(2, 2, wspace=0.03, hspace=0.03, width_ratios=(9, 1.5), height_ratios=(1.5, 9))
        else:
            grid_droite = global_grid[0, 1].subgridspec(2, 1, hspace=0.15, height_ratios=(5, 1))
            grid_droite_haut = grid_droite[0, 0].subgridspec(2, 2, wspace=0, hspace=0, width_ratios=(9, 1.5), height_ratios=(1.5, 9))
        ### Axes ###
        # Atribution des diférents axes a leurs positions sur la grille comme dédrit ci-dessus
        # Texte concernant les variables
        ax_titre_var = fig.add_subplot(global_grid_titre[0, 0])
        ax_titre_var.set_axis_off()
        # Les heatmaps:
        ax_heat_titre = fig.add_subplot(grid_gauche_bas_droite[0, 0])
        ax_heat_titre.set_axis_off()
        ## Contingence
        ax_heat_contin = fig.add_subplot(grid_gauche_bas_droite[1, 0])
        ax_heat_contin.set_title("Tableau de contingence")
        ## Fréquances relatives horisontales
        ax_heat_relh = fig.add_subplot(grid_gauche_bas_gauche[0, 0])
        ax_heat_relh.set_title("Fréquences partielles sur effectifs\nmarginaux horizontaux")
        ## Fréquances relatives verticales
        ax_heat_relv = fig.add_subplot(grid_gauche_bas_gauche[1, 0])
        ax_heat_relv.set_title("Fréquences partielles sur effectifs\nmarginaux verticaux")
        # Scatter et hist
        if (self.x_type != "QuantiContinu" and self.y_type != "QuantiContinu") or figure_droite == 'scatterplot':
            ax_scatter = fig.add_subplot(grid_droite_haut[1, 0])
            ax_hist_x = fig.add_subplot(grid_droite_haut[0, 0])
            ax_hist_y = fig.add_subplot(grid_droite_haut[1, 1])
        else:
            ax_scatter = fig.add_subplot(grid_droite_haut[1, 0])
            ax_hist_x = fig.add_subplot(grid_droite_haut[0, 0], sharex=ax_scatter)
            ax_hist_y = fig.add_subplot(grid_droite_haut[1, 1], sharey=ax_scatter)
        # Texte concernant les interactions des variables
        ax_text_var_inter = fig.add_subplot(grid_droite[1, 0])
        ax_text_var_inter.set_axis_off()
        
        ### Figures ###
        # Figures fixes
        ## Text descriptif des var ##
        ax_titre_var.text(0.02, 0.5, f"Analyses bivariées de {self.x.name} et de {self.y.name}:", fontsize=20, ha='left', va='center')
        inner_grid_text_var = grid_gauche[0, 0].subgridspec(1, 2, wspace=0.025, hspace=0.05)
        resources=[]
        if self.x_type[:5] == "Quant":
            resources.append((self.x.name, self.x_type, st.describe(self.x), test_norm(self.x, rtn="txt", name_distrib=False), "x"))
        else:
            contenue = ""
            if len(self.x.drop_duplicates()) > 5:
                contenue += ", ".join(list(self.x.drop_duplicates().sample(5)))[:25] + "..."
            else:
                contenue += ", ".join(list(self.x.drop_duplicates()))
            resources.append((self.x.name, self.x_type, self.x.describe(), contenue, "x"))
        
        if self.y_type[:5] == "Quant":
            resources.append((self.y.name, self.y_type, st.describe(self.y), test_norm(self.y, rtn="txt", name_distrib=False), "y"))
        else:
            contenue = ""
            if len(self.y.drop_duplicates()) > 5:
                contenue += ", ".join(list(self.y.drop_duplicates().sample(5)))[:25] + "..."
            else:
                contenue += ", ".join(list(self.y.drop_duplicates()))
            resources.append((self.y.name, self.y_type, self.y.describe(), contenue, "y"))
        
        for num_var in range(2):
            inner_grid_text_var_temp = inner_grid_text_var[0, num_var].subgridspec(3, 1, wspace=0.025, hspace=0.025, height_ratios=(2, 1, 4))
            ax_temp = fig.add_subplot(inner_grid_text_var[0, num_var])
            
            if resources[num_var][1][:5] == "Quant":
                ax_temp_var = fig.add_subplot(inner_grid_text_var_temp[0, 0])
                ax_temp_describe = fig.add_subplot(inner_grid_text_var_temp[1, 0])
                ax_temp_test = fig.add_subplot(inner_grid_text_var_temp[2, 0])
            else:
                ax_temp_var = fig.add_subplot(inner_grid_text_var_temp[0, 0])
                ax_temp_describe = fig.add_subplot(inner_grid_text_var_temp[1:, 0])
                
            fancybox = mpatches.FancyBboxPatch(
                [0.05, 0.05], 0.9, 0.9,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                alpha=0.4
                )
            ax_temp.add_patch(fancybox)
            ax_temp.set_axis_off()
            
            text_var = f"{resources[num_var][0] if len(resources[num_var][0])<lim_affiche_text else resources[num_var][0][:lim_affiche_text] +'...'} ({resources[num_var][4]}): {resources[num_var][1]}"
            ax_temp_var.text(0.065, 0.5, text_var, fontsize=12, ha='left', va='center', in_layout=True)
            ax_temp_var.set_axis_off()
            
            if resources[num_var][1][:5] == "Quant":
                text_describe = "  |  ".join([f"Interval : [{resources[num_var][2][1][0]}, {resources[num_var][2][1][1]}]",
                                            f"Moy= {format_numb(resources[num_var][2][2], 1)}"])
                ax_temp_describe.text(0.065, 0.5, text_describe, fontsize=10, ha='left', va='center', in_layout=True)
                ax_temp_describe.set_axis_off()
                
                text_test_brut = "".join(re.split(r"\[[a-z]*\]|\[/\]", resources[num_var][3][1]))
                text_test= "\n".join([f"test de normalitée de {resources[num_var][3][0]}:",
                                    f"{text_test_brut}"])
                ax_temp_test.text(0.065, 0.5, text_test, fontsize=10, ha='left', va='center', in_layout=True)
                ax_temp_test.set_axis_off()
            else:
                text_describe = "\n".join([f"Uniques : {format_numb(resources[num_var][2].iloc[1])}",
                                           f"Plus présent : {resources[num_var][2].iloc[2]} ({format_numb((resources[num_var][2].iloc[3]/resources[num_var][2].iloc[0]*100),2)}%)",
                                           f"Contenu : {resources[num_var][3]}"])
                ax_temp_describe.text(0.065, 0.6, text_describe, fontsize=10, ha='left', va='center', in_layout=True)
                ax_temp_describe.set_axis_off()

        
        ## Heatmapes ##
        # Titre de zone
        ax_heat_titre.text(0.05, 0.9, "Heatmaps:", fontsize=15, ha='left', va='center')
        # Tableau de contingence
        self.plot_heatmap(kind=ctg_heat_kind, ax=ax_heat_contin, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), cbar_kws=dict(shrink=cbar_hauteur), square=True, **kwargs_heatmap_ctg)
        ax_heat_contin.set_ylabel("")
        # Frec_h
        self.plot_heatmap(kind="frec_h", ax=ax_heat_relh, xticklabels=(), cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), cbar_kws=dict(shrink=cbar_hauteur), square=True, **kwargs_heatmap_par)
        ax_heat_relh.set_xlabel("")
        # Frec_v
        self.plot_heatmap(kind="frec_v", ax=ax_heat_relv, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), cbar_kws=dict(shrink=cbar_hauteur), square=True, **kwargs_heatmap_par)
        
        # Figures relatives aux types de variables
        ## Scatter et hist ##
        if self.x_type == "QuantiContinu" and self.y_type == "QuantiContinu":
            sns.histplot(x=self.x, ax=ax_hist_x, bins=15, **kwargs_histplot)
            sns.histplot(y=self.y, ax=ax_hist_y, bins=15, **kwargs_histplot)
        elif (self.x_type == "QuantiContinu" and self.y_type != "QuantiContinu") or figure_droite == 'boxplot':
            sns.histplot(y=self.x, ax=ax_hist_y, bins=15, **kwargs_histplot)
            sns.countplot(x=self.y, ax=ax_hist_x, **kwargs_countplot)
        elif (self.x_type != "QuantiContinu" and self.y_type == "QuantiContinu") or figure_droite == 'boxplot_inv':
            sns.countplot(y=self.x, ax=ax_hist_y, **kwargs_countplot)
            sns.histplot(x=self.y, ax=ax_hist_x, bins=15, **kwargs_histplot)
        elif self.x_type != "QuantiContinu" and self.y_type != "QuantiContinu":
            sns.countplot(y=self.x, ax=ax_hist_y, hue=self.x, palette="crest", **kwargs_countplot)
            sns.countplot(x=self.y, ax=ax_hist_x, hue=self.y, palette="crest", legend=False, **kwargs_countplot)
        ax_hist_x.set_axis_off()
        ax_hist_y.set_axis_off()
        
        pad = 25
        if (self.x_type == "QuantiContinu" and self.y_type == "QuantiContinu") or figure_droite == 'scatterplot':
            self.plot_scatter(ax=ax_scatter, **kwargs_scatterplot)
            ax_hist_x.set_title("Nuage de points et distributions respectives", pad=pad)
        elif figure_droite == 'boxplot':
            self.plot_box(ax=ax_scatter, **kwargs_boxplot)
            ax_hist_x.set_title("Boîtes à moustaches de chaques catégories", pad=pad)
        elif figure_droite == 'boxplot_inv':
            self.plot_box(ax=ax_scatter, inverse=True, **kwargs_boxplot)
            ax_hist_x.set_title("Boîtes à moustaches de chaques catégories", pad=pad)
        elif figure_droite == 'heatmap':
            self.plot_heatmap(kind="frec", pct=True, ax=ax_scatter, annot=True, cbar=False,
                         cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), linewidths=2, **kwargs_heatmap_quali)
            ax_hist_x.set_title("Heatmap des effectifs croisés (%)", pad=pad)
        elif (self.x_type == "QuantiContinu" and self.y_type != "QuantiContinu"):
            self.plot_box(ax=ax_scatter, **kwargs_boxplot)
            ax_hist_x.set_title("Boîtes à moustaches de chaques catégories", pad=pad)
        elif (self.x_type != "QuantiContinu" and self.y_type == "QuantiContinu"):
            self.plot_box(ax=ax_scatter, inverse=True, **kwargs_boxplot)
            ax_hist_x.set_title("Boîtes à moustaches de chaques catégories", pad=pad)
        elif (self.x_type != "QuantiContinu" and self.y_type != "QuantiContinu"):
            self.plot_heatmap(kind="frec", pct=True, ax=ax_scatter, annot=True, cbar=False,
                         cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), linewidths=2, **kwargs_heatmap_quali)
            ax_hist_x.set_title("Heatmap des effectifs croisés (%)", pad=pad)
        
        
        ## Corrélations ##
        ax_text_var_inter
        fancybox = mpatches.FancyBboxPatch(
            [0.03, 0.07], 0.85, 0.9,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02),
            alpha=0.4
            )
        ax_text_var_inter.add_patch(fancybox)
        if apparies is None:
            corr_test = "ATTENTION: renseigner si les variables sont appariées"
        else:
            corr_test = self.analyse_bivar_et_test(apparies=apparies, rtn_txt=True)
        corr_test_name = " ".join(corr_test[0].split("_"))
        try:
            corr_test_inter = "".join(re.split(r"\[[a-z]*\]|\[/\]", corr_test[1][1]))
            corr_test_inter = re.split('\n', corr_test_inter)
            corr_test_inter = corr_test_inter[:-1] + [corr_test_inter[-1][:int(len(corr_test_inter[-1])/2)] + '-\n    -' + corr_test_inter[-1][int(len(corr_test_inter[-1])/2):],]
            corr_test_inter = '\n'.join(corr_test_inter)
            text_describe = "\n".join([
                                   f"Stat= {format_numb(corr_test[1][0], 4)}",
                                   f"{corr_test_inter}"])
        except IndexError:
            text_describe = corr_test
        ax_text_var_inter.text(0.05, 0.9, f"{corr_test_name}:", fontsize=12, ha='left', va='top', in_layout=True)
        ax_text_var_inter.text(0.40, 0.9, text_describe, fontsize=10, ha='left', va='top', in_layout=True)
        
        # gestion des labeles d'echelles trop long
        for ax in fig.get_axes():
            y_ticklabels = ax.get_yticklabels()
            labels = [txt.get_text() if len(txt.get_text()) < limite_char_label else txt.get_text()[:limite_char_label-3] + "..." for txt in y_ticklabels]
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(labels)
        
        plt.show()
