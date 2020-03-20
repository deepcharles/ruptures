Documentation for the `ruptures` package.


In order to generate the documentation, the following packages are needed:
```
pip install sphinx
pip install sphinx_rtd_theme
pip install sphinxcontrib-bibtex
```

Make sure that the correct version of `ruptures` is installed. (The most probable situation is: you need the documentation for a local version of the `ruptures` library, therefore simply execute `python setup.py develop` in the top directory.)

Then, you only need to 
```
make clean
make html
```
