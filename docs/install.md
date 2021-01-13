# Installation

This library requires Python >=3.6 and the following packages: `numpy`, `scipy` and `matplotlib` (the last one is optional and only for display purposes).
You can either install the latest stable release or the development version.

## Stable release

To install the latest stable release, use `pip` or `conda`.

=== "With pip"
    ```
    python -m pip install ruptures
    ```

=== "With conda"
    `ruptures` can be installed from the `conda-forge` channel (run `conda config --add channels conda-forge` to add it):
    ```
    conda install ruptures
    ```

## Development release

Alternatively, you can install the development version of `ruptures` which can contain features that have not yet been integrated to the stable release.
To that end, refer to the [contributing guide](contributing.md).

## Upgrade

Show the current version of the package.

```
python -m pip show ruptures
```

In order to upgrade to the version, use the following command.

```
python -m pip install -U ruptures
```

