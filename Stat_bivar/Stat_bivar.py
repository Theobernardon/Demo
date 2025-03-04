import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
import re
import inspect
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
    "espace": "#1C5588 bold",
    "Erreur": "#e01616 bold"
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
def _edit_table_html(self, axe_x, axe_y, table_np, totaux=True):
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
            col_tab.sort_values()
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
    def _interpretation_cov(self, value, intencite=False):
        interpretation = ""
        x_name = self.x.name if len(self.x.name) < self.lim_affiche_text else self.x.name[:self.lim_affiche_text] + "..."
        y_name = self.y.name if len(self.y.name) < self.lim_affiche_text else self.y.name[:self.lim_affiche_text] + "..."
        if value < 0.0:
            interpretation = f"'{x_name}' et '{y_name}' évoluent dans des [motclef]directions opposées[/]"
            if intencite:
                interpretation += " " + self._test_d_intencitée(value)
        elif value > 0.0:
            interpretation = f"'{x_name}' et '{y_name}' évoluent dans le [motclef]même sens[/]"
            if intencite:
                interpretation += " " + self._test_d_intencitée(value)
        else:
            interpretation = f"'{x_name}' et '{y_name}' évoluent [motclef]indépendament l'un de l'autre[/]"
        return interpretation
    
    def _test_d_intencitée(value):
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
            liste_de_serie.append(self.df.loc[self.df.groupby(by=f"{quali.name}").groups[categ], f"{Quanti.name}"])
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
        juxtapose_html(contenue_html=[tb_initial, tb_attendue], 
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
            liste_de_serie.append(self.df.loc[self.df.groupby(by=f"{quali.name}").groups[categ], f"{quanti.name}"])
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
    
class StatBivar(BaseBivariateOutil):
    
    def __inti__(self, 
                 data: Union[pd.DataFrame, TableauContingence], 
                 colonne_x: Optional[str]=None, 
                 colonne_y: Optional[str]=None,
                 x_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 y_type: Literal["QuantiContinu", "QuantiDiscret", "QualiNominal", "QualiOrdinal", None]=None,
                 col_preformat_x: list=None, col_preformat_y: list=None,
                 nb_sep_QC = 10, nb_lim_QD = 10,
                 sep_round_x=0, sep_round_y=0):
        
        BaseBivariateOutil.__init__(self, data, 
                                    colonne_x, colonne_y, 
                                    x_type, y_type, 
                                    col_preformat_x, col_preformat_y, 
                                    nb_sep_QC, nb_lim_QD, 
                                    sep_round_x, sep_round_y)
        