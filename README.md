Algos de détection de ruptures:
====

Méthodes de parcours de l'espace des partitions
----
* Programmation dynamique. Fonctions de coût:
     * moyenne
     * variance
     * sinusoïde
     * noyaux


* PELT. Fonctions de coût:
     * moyenne
     * variance
     * sinusoïde
     * noyaux



* Régressions pénalisées. Pénalités:
    * fused lasso
    * fused rigde

Chaque algorithmes sera une classe avec une méthode "fit" pour calculer la segmentation. Certains paramètres seront dans le "init" et d'autres dans le "fit".
