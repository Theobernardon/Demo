# Stat_bivar

## Introduction

`Stat_bivar` est un package Python conçu pour effectuer des analyses statistiques bivariées. Il permet de réaliser des tests statistiques, de visualiser des données et de générer des tableaux de contingence. Ce package est particulièrement utile pour les analyses exploratoires de données et les tests d'hypothèses.

## Utilisation

### Importation du package

```python
from Stat_bivar import TableauContingence, TestStatBivar, StatBivarPlot
```

### Création d'un tableau de contingence

```python
import pandas as pd

# Chargement des données
df = pd.read_csv('path_to_your_data.csv')

# Création d'un tableau de contingence
tableau = TableauContingence(datframe=df, colonne_x='colonne_x', colonne_y='colonne_y')
```

### Affichage du tableau de contingence

```python
# L'affichage est automatique grâce à sa méthode _repr_html_
# Cette option ne renverra que le tableau de contingence classique
tableau
```
```python
# Cependant des déclinaisons sont possibles grâce à des méthodes
tableau.frequences()

tableau.frequances_partiels(axe="hv") # Horizontale et verticale

tableau.contingence_et_frequences()
```
```python
# Une méthode permettant de modifier les axes du tableau.
# Ceci permet de passer outre la gestion automatique des axes
tableau.set_axes(
    # Il suffit de lister les valeurs uniques dans l'ordre voulu
    axex=['1', '2', '3'], 
    # Ou de renseigner les valeurs d'intervalle voulu pour une variable continue : 
    # ATTENTION ]val1, val2] La première valeur est exclue !
    axey=[(-1, 5), (5, 10), (10, 20)] 
)
```

### Tests statistiques bivariés

```python
# Création d'un objet de test statistique
test_stat = TestStatBivar(data=df, colonne_x='colonne_x', colonne_y='colonne_y')
# ou 
test_stat = TestStatBivar(tableau)

# Test de normalité
test_stat.test_para(echantillon='x') # (Accès aux méthodes paramétriques ?)

# Test de corrélation de Pearson
test_stat.correlation_pearson()

# Test d'indépendance du Khi-deux
test_stat.test_independance_khi_deux()
```
La méthode la plus intéressante est est analyse_bivar_et_test. Cette méthode permet d'automatiser le choix du meilleur test statistique pour évaluer les 2 variables en fonction d'un arbre de décision.  
(Ceci étant une démo toutes les branches ne sont pas encore codées)
```python
test_stat.analyse_bivar_et_test(apparies=False)
```

### Visualisation des données

```python
import matplotlib.pyplot as plt

# Création d'un objet de visualisation
plot_stat = StatBivarPlot(data=df, colonne_x='colonne_x', colonne_y='colonne_y')
# ou 
plot_stat = TestStatBivar(tableau)

# Visualisation de la relation entre les variables
plot_stat.plot_rel()
plt.show()

# Visualisation de la heatmap
plot_stat.plot_heatmap()
plt.show()
```

### Analyse bivariée complète

```python
# Analyse bivariée complète avec visualisation
plot_stat.plot_annalyse_biv()
```

## Conclusion

Le package `Stat_bivar` offre une solution complète pour les analyses statistiques bivariées, incluant des tests d'hypothèses, des visualisations et des tableaux de contingence. Il est facile à utiliser et peut être intégré dans des notebooks Jupyter pour des analyses interactives.
