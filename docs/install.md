# Installation

This library requires Python3 and the following packages: `numpy`, `scipy` and `matplotlib` (the last one is optional and only for display purposes).
You can either install the latest stable release or the development version.

## Stable release

To install the latest stable release, use `pip` or `conda`.

=== "With pip"
    ```
    python -m pip install ruptures
    ```

=== "With conda"
    ```
    conda install ruptures
    ```

## Development release

Alternatively, you can install the development version of `ruptures` which can contain features that have not yet been integrated to the stable release.

Two methods are available: with `pip` and manually.
If you simply want the latest (and maybe untested) features, use `pip`.
In order to contribute to the library (bug fix, new feature, code or documentation improvement), please install manually from the Github repository.

=== "With pip"

    ```
    python -m pip install git+https://github.com/deepcharles/ruptures
    ```

=== "Manually"

    Start by cloning the Github repository.
    ```
    git clone https://github.com/deepcharles/ruptures
    cd ruptures
    ```
    Then install the downloaded package.
    ```
    python -m pip install --verbose --no-build-isolation --editable .
    ```

## Upgrade

Show the current version of the package.

```
python -m pip show ruptures
```

In order to upgrade to the version, use the following command.

```
python -m pip install -U ruptures
```

