Documentation for the `ruptures` package.


Avant de générer la documentation, il faut installer les packages suivants :
```
pip install sphinx
pip install sphinx_rtd_theme
pip install sphinxcontrib-bibtex
```

Sphinx utilise un `import ruptures` pour générer la documentation, donc il faut s'assurer que la bonne version de ruptures est importée par cette commande. Pour cela, on peut relancer un `python setup.py install develop` (en étant dans le bon repertoire).

Ensuite il suffit de 
```
make clean
make html
```
