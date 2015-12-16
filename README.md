#### Install
Après l'avoir téléchargé, dans le dossier du paquet:

> python3 setup.py develop

----
#### Algos de détection de ruptures:

Les methodes de détections de ruptures sont séparées selon la méthode de parcours de l'espace des partitions.

----
1. Programmation dynamique. Fonctions de coût:
    * [x] erreur quadratique, constant par morceaux
    * [x] erreur quadratique, constant par morceaux (version avec noyau)
    * [x] erreur quadratique, sinusoïde
    * [ ] erreur quadratique, linéaire par morceaux

2. PELT. Fonctions de coût:
    * [x] vraissemblance gaussienne (variance constante)
    * [ ] vraissemblance gaussienne (moyenne et variance variable)
    * [ ] erreur quadratique, constant par morceaux
    * [ ] erreur quadratique, linéaire par morceaux

3. Régressions pénalisées. Pénalités:
    * [ ] fused lasso (équivalent à PELT constant par morceau)
    * [ ] fused rigde
    * [x] pénalité L_0 (équivalent à PELT constant par morceau)

----
Chaque algorithme sera une classe avec une méthode "fit" pour calculer la segmentation.
Des exemples d'utilisation existent dans le dossier **tests/**

```python
import numpy as np
import ruptures.dynamic_programming.piecewise_constant as pc
n = 200
# 2 ruptures, 3 régimes
sig = np.array([1]*n+[0]*n+[1]*n)
# Calcul des ruptures
c = pc.Constant(sig)
res = c.fit(3, 1, 1)
ruptures = [s for (s, e) in res.keys() if s != 0]
ruptures.sort()
print(ruptures)
```
