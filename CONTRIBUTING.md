# Contributing


## Before contributing

In all following steps, it is highly recommended to use a virtual environnement.

### Install the development version

It is important that you contribute to the latest version of the code.
To that end, start by cloning the Github repository.

```
git clone https://github.com/deepcharles/ruptures
cd ruptures
```

Then install the downloaded package.

```
python -m pip install --verbose --no-build-isolation --editable .
```


### Install the requirements

Several packages are needed to format and test the code.
They are listed in `requirements-dev.txt` and can be installed with the following command.

=== "With pip"
    ```
    pip install -r requirements-dev.txt
    ```
=== "With conda"
    ```
    conda --file requirements-dev.txt
    ```

### Pre-commit hooks

We use `pre-commit` to run Git hooks before submitting the code to review.
These hook scripts perform simple tasks before each commit (code formatting
mostly).
Once it is installed (it is part of the [requirements](contributing.md#install-the-requirements)), run the following command.

```
pre-commit install
```

The list of hooks (and their options) can be found in [`.pre-commit-config.yaml`](https://github.com/deepcharles/ruptures/blob/master/.pre-commit-config.yaml).
For more information, see [their website](https://pre-commit.com/).

## Contribute to the code

### Write tests

### Write docstrings

## Contribute to the documentation

Use [MkDocs](https://www.mkdocs.org/).

Use `mkdocs serve` to preview your changes.
Once you are satisfied, no need to build the documentation, the CI will take care of that and publish it online at the next release of the package (if the pull request has been merged).

### Add examples to the gallery

An easy way to showcase your work with `ruptures` is to write a narrative example.
To that, simply put a [Jupyter notebook](https://jupyter.org/) in the `notebooks/` folder.
To make it appear in the documentation, add a reference in `mkdocs.yml` (`nav > Gallery of examples`): if the notebook's name is `my_notebook.ipynb`, it will be available as `notebooks/my_notebook.md`.
It will be rendered automatically when [MkDocs](https://www.mkdocs.org/) builds the documentation.

We welcome any interesting work about a new cost function, algorithm, data, calibration method, etc.
Any other package can be used in combination with `ruptures`.
However, each example should be clearly explained with text and figures.
The amount of raw code should also remain limited for readability.


## Miscellaneous

### Naming convention

We try to follow (roughly) a consistent naming convention of modules, classes, functions, etc.
When in doubt, you can refer to the [PEP 8 style guide for Python code](https://www.python.org/dev/peps/pep-0008/#naming-conventions).