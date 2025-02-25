**Tutoriel sur la classe `stat_univariee`**
=====================================================

**Introduction**
---------------

La classe `stat_univariee` est une classe Python qui permet de réaliser des analyses univariées de données. Elle est conçue pour être utilisée avec des données Pandas. L'objectif est d'avoir tout à portée de main pour les analyses de première intentions sans négliger l'aspect visuel pour autant. Une interface codée en HTML permet de disposer des différentes analyses universées des colonnes d'un dataframe sur un même espace.

**Création d'une instance de `stat_univariee`**
---------------------------------------------

Pour créer une instance de `stat_univariee`, il suffit de passer les arguments suivants :

* `datafram` : un DataFrame ou une série Pandas contenant les données à analyser.
* `hue` : une série Pandas catégorielle optionnelle. Si cette variable est utilisée, le dataframe ou la série précédente sera décomposé en fonction de chaque catégorie. L'analyse univariée se fera sur chacune de ces catégories puis sera restitué sur la même figure.

```python
import seaborn as sns
from stat_univar import stat_univariee

# Charger le dataset Titanic intégré dans seaborn
df = sns.load_dataset('titanic')

# Création d'une instance de stat_univariee
fig = stat_univariee(df.loc[:, ['age', 'fare', 'embark_town']])
fig
```

```python
# Création d'une instance de stat_univariee par catégorie 'sex'
fig = stat_univariee(df.loc[:, ['age', 'fare', 'embark_town']], hue=df['sex'])
fig
```

**Méthodes de la classe `stat_univariee`**
-----------------------------------------

La classe `stat_univariee` s'utilise principalement au moment de son instanciation. L'objet sera là pour pouvoir ré-afficher La figure ou la sauvegarder dans un fichier HTML:

* `_repr_html_` : méthode qui affiche l'analyse univariée en HTML.   
(Cette méthode est automatiquement appelée dans un Jupiter notebook)

```python
# Affichage de la figure via la méthode _repr_html_()
fig 
```
```python
# Sauvegarde de la figure sous un fichier HTML
fig.save_html()
```

**Personnalisation de l'analyse univariée**
-----------------------------------------

La classe `stat_univariee` offre plusieurs paramètres pour personnaliser l'analyse univariée :

* `log_scale` : paramètre qui définit si l'échelle de l'analyse est logarithmique ou non.
* `hist_bins` : paramètre qui définit le nombre de séparateurs sur l'histogramme.
* `lorenz` : paramètre qui définit si la courbe de Lorenz est affichée ou non.
* `lorenz_x(y)ticks` : Paramètres qui rajoutent des lignes de marquage sur les axes X ou y permettant de mettre en avant des valeurs importantes.
* `figsize` : paramètre qui définit la taille de la figure.
* `text_frontsize` : paramètre qui définit la taille de la police.
* `stat_centrales_lines` : paramètre qui définit si les lignes de stats de tendances sont affichées ou non.

```python
# Exemple de personnalisation de l'analyse univariée
stat = stat_univariee(df, hue=df['categorie'], log_scale=True, hist_bins=50, lorenz=False)
stat
```
