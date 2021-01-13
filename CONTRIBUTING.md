# Contributing


## Before contributing

In all following steps, it is highly recommended to use a virtual environment.
Build and installation are performed using `pip` so be sure to have the latest version available.

```
python -m pip install --upgrade pip
```

### Install the development version

It is important that you contribute to the latest version of the code.
To that end, start by cloning the Github repository.

```
git clone https://github.com/deepcharles/ruptures
cd ruptures
```

Then install the downloaded package with `pip`.

```
python -m pip install --editable .[dev]
```

Note that `python -m` can be omitted most of the times, but within virtualenvs, it can prevent certain errors.
Also, in certain terminals (such as `zsh`), the square brackets must be escaped, e.g. replace `.[dev]` by `.\[dev\]`.

In addition to `numpy`, `scipy` and `ruptures`, this command will install all packages needed to develop `ruptures`.
The exact list of librairies can be found in the [`setup.cfg` file](https://github.com/deepcharles/ruptures/blob/master/setup.cfg) (section `[options.extras_require]`).

### Pre-commit hooks

We use `pre-commit` to run Git hooks before submitting the code to review.
These hook scripts perform simple tasks before each commit (code formatting mostly).
To activate the hooks, simply run the following command in your terminal.

```
pre-commit install
```

If you try to commit a non-compliant (i.e. badly formatted) file, `pre-commit` will modify this file and make the commit fail.
However you need to stage the new changes **yourself** as `pre-commit` will not do that for you (this is by design; see [here](https://github.com/pre-commit/pre-commit/issues/806) or [here](https://github.com/pre-commit/pre-commit/issues/747)).
Fortunately, `pre-commit` outputs useful messages.

The list of hooks (and their options) can be found in [`.pre-commit-config.yaml`](https://github.com/deepcharles/ruptures/blob/master/.pre-commit-config.yaml).
For more information, see [their website](https://pre-commit.com/).
If you want to manually run all pre-commit hooks on a repository, run `pre-commit run --all-files`. To run individual hooks use `pre-commit run <hook_id>`.

## Contribute to the code

### Write tests

The following command executes the test suite.

```
python -m pytest
```

### Write docstrings

## Contribute to the documentation

Use [MkDocs](https://www.mkdocs.org/).

Use `mkdocs serve` to preview your changes.
Once you are satisfied, no need to build the documentation, the CI will take care of that and publish it online at the next release of the package (if the pull request has been merged).

### Add examples to the gallery

An easy way to showcase your work with `ruptures` is to write a narrative example.
To that, simply put a [Jupyter notebook](https://jupyter.org/) in the `docs/examples` folder.
To make it appear in the documentation, add a reference in `mkdocs.yml` (`nav > Gallery of examples`): if the notebook's name is `my_notebook.ipynb`, it will be available as `examples/my_notebook.ipynb`.
It will be rendered automatically when [MkDocs](https://www.mkdocs.org/) builds the documentation.

!!! note
    To automatically add a [Binder](https://mybinder.org/v2/gh/deepcharles/ruptures/master) link and a download link to your notebook, simply add the following line of code.
    ```markdown
    {{ '<!-- {{ add_binder_block(page) }} -->' }}
    ```
    Ideally, place this code below the title of the notebook (same cell) and it will be rendered as in [here](examples/kernel-cpd-performance-comparison.ipynb).

We welcome any interesting work about a new cost function, algorithm, data, calibration method, etc.
Any other package can be used in combination with `ruptures`.
However, each example should be clearly explained with text and figures.
The amount of raw code should also remain limited for readability.


## Miscellaneous

### Naming convention

We try to follow (roughly) a consistent naming convention of modules, classes, functions, etc.
When in doubt, you can refer to the [PEP 8 style guide for Python code](https://www.python.org/dev/peps/pep-0008/#naming-conventions).
