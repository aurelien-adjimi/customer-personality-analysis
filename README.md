# customer-personality-analysis

# Veille sur la classification non supervisée  
## Introduction  
### Présentation du projet  
Ce projet, que nous réalisons en trio, va nous permettre de mettre en oeuvre des algorithmes de Classification Non Supervisée. Pour cela, nous allons devoir réaliser un regroupement non supervisé de clients à partir des données d'une petite épicerie. Nous allons devoir appliquer trois algorithmes de classification non supervisée différents afin de déterminer lequel s'applique au mieu à notre de jeu de données, que nous aurons préalablement nettoyées et préparées.  

### Contexte  
L'analyse de la personnalité du client est une analyse détaillée d’un groupe de clients idéaux pour une entreprise. Cela aide les entreprises à mieux comprendre leur clientèle et leur permet de modifier leurs produits ainsi que leurs
approches marketing de manière plus efficace et plus adaptée à leur clientèle.  
Cette analyse est très importante pour le développement d’une entreprise car elle se fait en fonction des besoins, des comportements et des préoccupations des différents types de segments de clients.  

### Déroulement  
Dans un premier temps, nous devons réaliser deux veilles:  
- La première sur les algorithmes de classification non supervisée. On doit apprendre au moins trois algorithmes différents.  
- La deuxième sur les méthodes de sélection du nombre optimal de clusters et la mesure de qualité d’un cluster.  

En suite, nous devons implémenter notre propre Class "K-means", ce qui veut dire que nous devons créer notre propre algorithme puis dans un second temps faire la même chose en appliquant la fonction K-means de Scikit-Learn au dataset Iris.  
Une fois que ceci est fait, nous devons lancer les modèles plusieurs fois et pour plusieurs valeurs de k. En suite, nous devons évaluer le nombre optimal de clusters ainsi que leur qualité.  

Une fois que tout ceci est fait, nous pouvons passer aux données de l'épicerie de notre quartier. Nous devorns donc:  
- Explorer et analyser les données  
- Nettoyer et pré-traiter les données  
- Réaliser une réduction de dimension à l’aide de la sélection de feature et/ou l'analyse à facteurs multiples  
- Appliquer trois algorithmes de Classification Non Supervisée  
- Comparer les résultats et évaluer les modèles  
- Définir ce qui caractérise les individus de chaque groupe  
- Conclure sur nos résultats.  


## Le jeu de données  
### Présentation des données  

### Préparation des données  

### Analyse des données  

## Les différents algorithmes de Classification Non Supervisée  
### Veille sur la Classification Non supervisée  
La classification à pour but de regrouper _n_ observations en un certains nombre de groupes ou de classes homogènes. Comme nous le savons déjà, il existe de types principaux de classification:  
- Supervisée  
- Non Supervisée  

Nous allons ici nous pencher sur la Classification Non Supervisée.  
En classification non supervisée, on ne connait pas le nombre de groupes qui existent dans la population, on ne connait pas le groupe auquel appartient chaque observation de la population et on veut classer les observations dans des groupes homogènes à partir de différentes variables.  
Parmi les applications typiques les plus connues on retrouve:  
- Biologie, pour l'élaboration de la taxonomie animale.  
- Psychologie, pour la détermination des types de personnalités présents dans un groupe d'individus.  
- Text Mining, pour le partitionnement de courriels ou textes en fonction du sujet traité.  

Il existe plusieurs familles de méthodes de classification non supervisée. Les plus communes sont:  

la classification hiérarchique;  
la classification non hiérarchique, par exemple la méthode des k-moyennes (k-means);  
la classification basée sur une densité;  
la classification basée sur des modèles statistiques/probabilistes, par exemple un mélange de lois normales.  

Pour regrouper des observations en groupes homogènes, il faut tout d’abord avoir une définition de ce que sont des observations similaires ou des observations différentes. Il faut donc être en mesure de quantifier la similarité ou la distance entre deux observations. Cette première étape peut parfois être la plus difficile de tout le processus de classification, mais elle est essentielle et est le premier pas de toute analyse de partitionnement.  

Si les observations sont constituées de _p_ nombres réels de valeurs du même ordre de grandeur, alors la distance euclidienne entre les deux vecteurs dans $\mathbb{R}^p$ est une mesure tout à fait raisonnable. Mais comment faire lorsque les observations sont constituées de _p_ variables binaires (oui/non, homme/femme...), ou _p_ variables catégorielles, des images, des textes où même un mélange de tout cela ?  
Plusieurs mesures ont été développées sur mesure pour leur application particulière à force d'expérience et expérimentation. C'est ce que nous allons voir dans la suite de cette veille.  

**Mesure de distance:**  
Une mesure de distance _d_ doit satisfaire les propriétés suivantes pour tout _i_, _j_, _k_ $\in$ {1, . . . , _n_}:  
- _d_(_i_, _j_) $\geq$ 0;  
- _d_(_i_, _j_) = 0;  
- _d_(_i_, _j_) = _d_(_j_, _i_);  
- _d_(_i_, _k_) $\leq$ _d_(_i_, _j_) + _d_(_j_, _k_).  

La distance $\lambda _q_$ entre deux vecteurs dans $\mathbb{R}^p$ est définie par:  
||_x_$_i$ - _x_$_j$||$_q$ = ($\sum_{k = 1}^p$ |x$_ik$ - x$_jk$|^q)^$\frac{1}{q}$.  
La distance euclidienne correspond au cas où q = 2.  

La distance $\lambda _q_$ n'est pas invariante à un changement d'échelle.Ce qui a des conséquences majeures pour la pratique.  
Par exemple, considérons le jeu de données suivants:  
| Poids en grammes | Taille en centimètres |
| :- | -: |
| 10 | 7 |
| 20 | 2 |
| 30 | 10 |  

On trouve les distances suivantes:  
_d_(1, 2) = 11.2, _d_(1,3) = 20.2, _d_(2,3) = 12.8  

Si la taille est exprimée en millimètres, on trouve les distances suivantes:  
_d_(1, 2) = 51.0, _d_(1,3) = 36.1, _d_(2,3) = 80.6  

On peut donc se demander si le premier objet est plus près du deuxième objet ou du troisième objet ?  
Cet exemple explique pourquoi dans plusieurs situations on préfère travailler avec la distance standardisée entre les variables,  

_d²_(x$_i$, x$_j$) = $\sum{k=1}^_p$($\frac{x_ik - µ_k}{s_k} - \frac{x_jk - µ_k}{s_k}$)² = $\sum{k=1}^p$($\frac{x_ik - x_jk}{s_k}$)²  

Où  

µ_k = moyenne de la variable k; s_k = écart type de la variable k.  

On trouve les distances suivantes, peu importe l'unité de mesure utilisée:  
_d_(1, 2) = 16, _d_(1, 3) = 42, _d_(2, 3) = 26.

### Algo 1  

### Algo 2  

### Algo 3  

## Les Clusters  

### Méthodes de sélection du nombre optimal de clusters  

### Mesure de qualité d'un cluster  




## Conclusion
